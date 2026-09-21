import unittest
import torch
from torch.utils.data import TensorDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from src.train.RFT.resume_prefetch import rewind_stateless_prefetch


class ResumePrefetchTest(unittest.TestCase):
    def test_shuffled_order_is_preserved(self):
        dataset = TensorDataset(torch.arange(12))
        loader = StatefulDataLoader(dataset, batch_size=1, num_workers=2, shuffle=True,
                                    generator=torch.Generator().manual_seed(42))
        iterator = iter(loader)
        consumed = [int(next(iterator)[0].item()) for _ in range(5)]
        original = loader.state_dict()
        expected = consumed[-1:] + [int(x[0].item()) for x in iterator]
        resumed = StatefulDataLoader(dataset, batch_size=1, num_workers=2, shuffle=True)
        resumed.load_state_dict(original)
        resumed.load_state_dict(rewind_stateless_prefetch(resumed.state_dict(), 4))
        self.assertEqual([int(x[0].item()) for x in resumed], expected)
        self.assertEqual(original['_snapshot']['_main_snapshot']['_sampler_iter_state']
                         ['sampler_iter_state']['yielded'], 5)

    def test_next_unconsumed_batch_is_replayed(self):
        dataset = TensorDataset(torch.arange(12))
        loader = StatefulDataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)
        iterator = iter(loader)
        for _ in range(5):
            next(iterator)
        original = loader.state_dict()
        resumed = StatefulDataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)
        resumed.load_state_dict(original)
        resumed.load_state_dict(rewind_stateless_prefetch(resumed.state_dict(), 4))
        self.assertEqual([int(x[0].item()) for x in resumed], list(range(4, 12)))
        self.assertEqual(original['_snapshot']['_snapshot_step'], 5)
        with self.assertRaises(AssertionError):
            rewind_stateless_prefetch(original, 3)
