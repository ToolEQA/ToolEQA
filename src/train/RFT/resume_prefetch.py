"""Recover an unconsumed one-step-prefetched batch without editing checkpoint files."""
import copy


def rewind_stateless_prefetch(state, completed_steps):
    state = copy.deepcopy(state)
    snap = state['_snapshot']
    main = snap['_main_snapshot']
    expected = completed_steps + 1
    assert state['_steps_since_snapshot'] == 0 and not state['_iterator_finished']
    assert snap['_snapshot_step'] == main['_sampler_iter_yielded'] == expected
    sampler = main['_sampler_iter_state']
    assert sampler['samples_yielded'] == expected
    assert set(sampler) <= {'samples_yielded', 'sampler_iter_state'}
    if 'sampler_iter_state' in sampler:
        random_state = sampler['sampler_iter_state']
        assert set(random_state) == {'yielded', 'generator'}
        assert random_state['yielded'] == expected
        random_state['yielded'] -= 1
    assert main['_index_sampler_state'] is None
    assert all(w['dataset_state'] is None and w['fetcher_state'] is None
               for w in snap['_worker_snapshots'].values())
    snap['_snapshot_step'] -= 1
    main['_sampler_iter_yielded'] -= 1
    main['_sampler_iter_state']['samples_yielded'] -= 1
    snap['_last_yielded_worker_id'] = (snap['_last_yielded_worker_id'] - 1) % main['_num_workers']
    return state
