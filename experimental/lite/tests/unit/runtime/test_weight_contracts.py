import pytest

from megatron.lite.runtime.contracts.weights import ResyncFormat


@pytest.mark.parametrize(
    "expected",
    [ResyncFormat.BF16, ResyncFormat.BLOCK_FP8, ResyncFormat.MXFP4],
)
def test_resync_format_round_trip(expected: ResyncFormat) -> None:
    assert ResyncFormat.parse(expected.value) is expected
    assert ResyncFormat.parse(expected) is expected


def test_resync_format_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="mxfp4"):
        ResyncFormat.parse("fp4")
