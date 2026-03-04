# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import logging
import typing

import pytest
import torch

try:
    from torch.distributed import DeviceMesh
    from torch.distributed._tensor import DTensor

    HAVE_DTENSOR = True
except ImportError:
    HAVE_DTENSOR = False

pytest.importorskip(
    "nvidia_resiliency_ext", reason="MCoreTensorAwareStateDict requires nvidia-resiliency-ext"
)

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.dist_checkpointing.dict_utils import merge
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.dist_checkpointing.tensor_aware_state_dict import MCoreTensorAwareStateDict
from megatron.core.dist_checkpointing.validation import StrictHandling
from tests.unit_tests.test_utilities import Utils


class TestStrictLocal:
    def setup_method(self, method):
        Utils.initialize_model_parallel(8, 1)  # doesn't matter for this test

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _get_base_state_dict(self):
        return {
            'TenA': ShardedTensor.from_rank_offsets('TenA', torch.arange(2), replica_id=Utils.rank),
            'TenB': ShardedTensor.from_rank_offsets(
                'TenB', torch.arange(3), (0, Utils.rank, Utils.world_size), replica_id=0
            ),
            'TenC': ShardedTensor.from_rank_offsets(
                'TenC', torch.arange(3), replica_id=Utils.world_size - Utils.rank - 1
            ),
            'ObjA': ShardedObject('ObjA', list(range(10)), (1,), (0,), replica_id=Utils.rank),
            'ObjB': ShardedObject(
                'ObjB', {Utils.rank + 7}, (1, Utils.world_size), (0, Utils.rank), replica_id=0
            ),
            'Nested': {
                'TenE': ShardedTensor.from_rank_offsets(
                    'Nested.TenE', torch.arange(3), replica_id=Utils.world_size - Utils.rank - 1
                ),
                'ObjE': ShardedObject(
                    'Nested.ObjE', list(range(10)), (1,), (0,), replica_id=Utils.rank
                ),
                'TenF': ShardedTensor.from_rank_offsets(
                    'Nested.TenF', torch.arange(3), replica_id=Utils.rank
                ),
                'ObjF': ShardedObject(
                    'Nested.ObjF', list(range(10)), (1,), (0,), replica_id=Utils.rank
                ),
            },
            'NestedEmpty': {},
        }

    def _get_extra_state_dict(self):
        return {
            'UnexpectedTenD': ShardedTensor.from_rank_offsets(
                'UnexpectedTenD', torch.arange(3), replica_id=Utils.rank
            ),
            'UnexpectedObjD': ShardedObject(
                'UnexpectedObjD', None, (1,), (0,), replica_id=Utils.rank
            ),
            'UnexpectedNested': {
                'UnexpectedTenF': ShardedTensor.from_rank_offsets(
                    'UnexpectedNested.UnexpectedTenF', torch.arange(3), replica_id=Utils.rank
                ),
                'UnexpectedObjF': ShardedObject(
                    'UnexpectedNested.UnexpectedObjF', None, (1,), (0,), replica_id=Utils.rank
                ),
            },
            'Nested': {
                'UnexpectedTenG': ShardedTensor.from_rank_offsets(
                    'Nested.UnexpectedTenG', torch.arange(3), replica_id=Utils.rank
                ),
                'UnexpectedObjG': ShardedObject(
                    'Nested.UnexpectedObjG', None, (1,), (0,), replica_id=Utils.rank
                ),
            },
            'NestedEmpty': {
                'UnexpectedTenH': ShardedTensor.from_rank_offsets(
                    'NestedEmpty.UnexpectedTenH', torch.arange(3), replica_id=Utils.rank
                ),
                'UnexpectedObjH': ShardedObject(
                    'NestedEmpty.UnexpectedObjH', None, (1,), (0,), replica_id=Utils.rank
                ),
            },
        }

    def _tasd_to_state_dict(self, *, algo, strict, validate_access_integrity, missing, unexpected):
        sharded_state_dict = self._get_base_state_dict()
        if missing:
            del sharded_state_dict['TenA']
            del sharded_state_dict['ObjB']
            del sharded_state_dict['Nested']['TenE']
            del sharded_state_dict['Nested']['ObjF']
            del sharded_state_dict['NestedEmpty']
        if unexpected:
            # Note: merge is in-place
            sharded_state_dict = merge(sharded_state_dict, self._get_extra_state_dict())
        tasd, _ = MCoreTensorAwareStateDict.from_state_dict(self._get_base_state_dict(), algo)
        tasd = typing.cast(MCoreTensorAwareStateDict, tasd)
        return tasd.to_state_dict(
            sharded_state_dict=sharded_state_dict,
            validate_access_integrity=validate_access_integrity,
            strict=strict,
            algo=algo,
            return_mismatch_keys=True,
        )

    @property
    def _missing_keys(self):
        return {'TenA', 'ObjB', 'Nested.TenE', 'Nested.ObjF'}

    @property
    def _unexpected_keys(self):
        return {
            'UnexpectedTenD',
            'UnexpectedObjD',
            'UnexpectedNested.UnexpectedTenF',
            'UnexpectedNested.UnexpectedObjF',
            'NestedEmpty.UnexpectedTenH',
            'NestedEmpty.UnexpectedObjH',
        }

    def _check_log_message(self, text, should_contain_missing, should_contain_unexpected):
        if not should_contain_missing and not should_contain_unexpected:
            assert text == ""
            return
        if should_contain_missing:
            assert 'Missing keys' in text
            for key in self._missing_keys:
                assert key in text
        else:
            assert 'Missing keys' not in text
            for key in self._missing_keys:
                assert key not in text
        if should_contain_unexpected:
            assert 'Unexpected keys' in text
            for key in self._unexpected_keys:
                assert key in text
        else:
            assert 'Unexpected keys' not in text
            for key in self._unexpected_keys:
                assert key not in text

    def _check_log_message_for_strict_handling(self, text, strict, missing, unexpected):
        # Answers the question:
        # "I got the log message [text] using strictness [strict]. I [removed/didn't remove] missing and [added/didn't add] unexpected keys."
        # Is the log correct?
        should_contain_unexpected = (
            strict in {StrictHandling.LOG_UNEXPECTED, StrictHandling.LOG_ALL}
        ) and unexpected
        should_contain_missing = (strict in {StrictHandling.LOG_ALL}) and missing
        return self._check_log_message(text, should_contain_missing, should_contain_unexpected)

    def _check_return_values(
        self, missing_keys, unexpected_keys, should_contain_missing, should_contain_unexpected
    ):
        if should_contain_missing:
            assert set(missing_keys) == self._missing_keys
        else:
            assert set(missing_keys) == set()
        if should_contain_unexpected:
            assert set(unexpected_keys) == self._unexpected_keys
        else:
            assert set(unexpected_keys) == set()

    def _check_return_values_for_strict_handling(
        self, strict, missing_keys, unexpected_keys, missing, unexpected
    ):
        should_contain_missing = (
            strict in {StrictHandling.RETURN_ALL, StrictHandling.LOG_ALL}
        ) and missing
        should_contain_unexpected = (
            strict
            in {
                StrictHandling.RETURN_ALL,
                StrictHandling.RETURN_UNEXPECTED,
                StrictHandling.LOG_UNEXPECTED,
                StrictHandling.LOG_ALL,
            }
        ) and unexpected
        self._check_return_values(
            missing_keys, unexpected_keys, should_contain_missing, should_contain_unexpected
        )

    @pytest.mark.parametrize('algo', ['fully_parallel', 'atomic'])
    @pytest.mark.parametrize('validate_integrity', [True, False])
    @pytest.mark.parametrize('strict', list(StrictHandling))
    def test_everything_ok(self, caplog, algo, validate_integrity, strict):
        with caplog.at_level(logging.WARNING):
            state_dict, missing_keys, unexpected_keys = self._tasd_to_state_dict(
                algo=algo,
                strict=strict,
                validate_access_integrity=validate_integrity,
                missing=False,
                unexpected=False,
            )
            assert state_dict.keys() == self._get_base_state_dict().keys()
            assert set(missing_keys) == set()
            assert set(unexpected_keys) == set()
        assert caplog.text == ''

    @pytest.mark.parametrize('algo', ['atomic'])
    @pytest.mark.parametrize('validate_integrity', [True, False])
    @pytest.mark.parametrize(['missing', 'unexpected'], [(True, False)])
    @pytest.mark.parametrize(
        'strict',
        [
            StrictHandling.ASSUME_OK_UNEXPECTED,
            StrictHandling.LOG_UNEXPECTED,
            StrictHandling.LOG_ALL,
            StrictHandling.RETURN_UNEXPECTED,
            StrictHandling.RETURN_ALL,
            StrictHandling.IGNORE_ALL,
        ],
    )
    def test_passthrough(self, caplog, algo, validate_integrity, missing, unexpected, strict):
        # Scenario: strictness check is supposed to pass the errors through, the underlying algorithm is able to handle it.
        with caplog.at_level(logging.WARNING):
            _, missing_keys, unexpected_keys = self._tasd_to_state_dict(
                algo=algo,
                strict=strict,
                validate_access_integrity=validate_integrity,
                missing=missing,
                unexpected=unexpected,
            )
        self._check_log_message_for_strict_handling(caplog.text, strict, missing, unexpected)
        self._check_return_values_for_strict_handling(
            strict, missing_keys, unexpected_keys, missing, unexpected
        )

    # NOTE: Fully parallel results in a hard-to-catch error:
    #       The exchange algorithm is unaware of the missing tensors and will still expect the shards to be received -
    #       which will cause the process to hang indefinitely.
    @pytest.mark.parametrize('algo', ['atomic'])
    @pytest.mark.parametrize('validate_integrity', [True, False])
    @pytest.mark.parametrize(['missing', 'unexpected'], [(False, True), (True, True)])
    @pytest.mark.parametrize(
        'strict',
        [
            StrictHandling.ASSUME_OK_UNEXPECTED,
            StrictHandling.LOG_UNEXPECTED,
            StrictHandling.LOG_ALL,
            StrictHandling.RETURN_UNEXPECTED,
            StrictHandling.RETURN_ALL,
            StrictHandling.IGNORE_ALL,
        ],
    )
    def test_passthrough_errors(
        self, caplog, algo, validate_integrity, missing, unexpected, strict
    ):
        # Scenario: strictness check is supposed to pass the errors through,
        # but they result in an error in the underlying algorithm as it's unable to handle it.
        # That's why "Fully parallel" is excluded, as instead of raising an error, it will hang indefinitely, which is hard to catch.
        with caplog.at_level(logging.WARNING):
            with pytest.raises(AssertionError) as exc_info:
                self._tasd_to_state_dict(
                    algo=algo,
                    strict=strict,
                    validate_access_integrity=validate_integrity,
                    missing=missing,
                    unexpected=unexpected,
                )
            # TODO: check exc_info
        self._check_log_message_for_strict_handling(caplog.text, strict, missing, unexpected)

    @pytest.mark.parametrize('algo', ['fully_parallel', 'atomic'])
    @pytest.mark.parametrize('validate_integrity', [True, False])
    @pytest.mark.parametrize('missing', [True, False])
    def test_raise_unexpected(self, validate_integrity, algo, missing):
        with pytest.raises(CheckpointingException) as exc_info:
            self._tasd_to_state_dict(
                algo=algo,
                strict=StrictHandling.RAISE_UNEXPECTED,
                validate_access_integrity=validate_integrity,
                missing=missing,
                unexpected=True,
            )
        self._check_log_message(
            str(exc_info.value), should_contain_missing=False, should_contain_unexpected=True
        )

    @pytest.mark.parametrize('algo', ['fully_parallel', 'atomic'])
    @pytest.mark.parametrize('validate_integrity', [True, False])
    @pytest.mark.parametrize(
        ['missing', 'unexpected'], [(True, False), (False, True), (True, True)]
    )
    def test_raise_all(self, validate_integrity, algo, missing, unexpected):
        with pytest.raises(CheckpointingException) as exc_info:
            self._tasd_to_state_dict(
                algo=algo,
                strict=StrictHandling.RAISE_ALL,
                validate_access_integrity=validate_integrity,
                missing=missing,
                unexpected=unexpected,
            )
        self._check_log_message(
            str(exc_info.value),
            should_contain_missing=missing,
            should_contain_unexpected=unexpected,
        )
