"""Compatibility entry point for the controller-only SFT converter.

Like convert_data_to_qwen, convert_sample_to_qwen_style returns a list of
per-turn examples. Both commands accept --input, --output, and --limit.
"""

if __package__:
    from .convert_data_to_qwen import convert_sample_to_qwen_style, main
else:
    from convert_data_to_qwen import convert_sample_to_qwen_style, main


if __name__ == "__main__":
    main()
