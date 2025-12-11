def compare_helpers(pipeline_parallel_size, num_microbatches, num_model_chunks):
    total_num_microbatches = num_microbatches * num_model_chunks

    # Baseline helpers
    def baseline_get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (pipeline_parallel_size * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def baseline_get_microbatch_id_in_model_chunk(iteration_id, forward):
        """Helper method to get the microbatch_id within model chunk given the iteration number."""
        assert forward
        iteration_group_id = iteration_id // (pipeline_parallel_size * num_model_chunks)
        microbatch_id_in_model_chunk = (iteration_group_id * pipeline_parallel_size) + (
            iteration_id % pipeline_parallel_size
        )
        return microbatch_id_in_model_chunk

    def baseline_is_first_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == 0:
            return microbatch_id_in_group % pipeline_parallel_size == 0
        else:
            return False

    def baseline_is_last_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == num_microbatch_groups - 1:
            return microbatch_id_in_group % pipeline_parallel_size == pipeline_parallel_size - 1
        else:
            return False

    # Create schedule table prior to new helper methods
    schedule_table = []
    for min_microbatch_id_in_group in range(0, num_microbatches, pipeline_parallel_size):
        if min_microbatch_id_in_group + pipeline_parallel_size >= num_microbatches:
            # Construct schedule for the last microbatch group
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(num_model_chunks)
                    for microbatch_id in range(min_microbatch_id_in_group, num_microbatches)
                ]
            )
        else:
            # Construct schedule for other microbatch groups
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(num_model_chunks)
                    for microbatch_id in range(
                        min_microbatch_id_in_group,
                        min_microbatch_id_in_group + pipeline_parallel_size,
                    )
                ]
            )

    microbatch_id_table, model_chunk_id_table = zip(*schedule_table)

    # New helper methods that indexes schedule table
    def new_get_model_chunk_id(virtual_microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        model_chunk_id = model_chunk_id_table[virtual_microbatch_id % total_num_microbatches]
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def new_get_microbatch_id_in_model_chunk(iteration_id, forward):
        """Helper method to get the microbatch_id within model chunk given the iteration number."""
        assert forward
        microbatch_id_in_model_chunk = microbatch_id_table[iteration_id]
        return microbatch_id_in_model_chunk

    def new_is_first_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        if virtual_microbatch_id < total_num_microbatches:
            return microbatch_id_table[virtual_microbatch_id] == 0
        else:
            return False

    def new_is_last_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        if virtual_microbatch_id < total_num_microbatches:
            return microbatch_id_table[virtual_microbatch_id] == num_microbatches - 1
        else:
            return False

    for i in range(total_num_microbatches):
        # Test both forward and backward
        assert baseline_get_model_chunk_id(i, forward=False) == new_get_model_chunk_id(
            i, forward=False
        )
        assert baseline_get_model_chunk_id(i, forward=True) == new_get_model_chunk_id(
            i, forward=True
        )

        # Only used in forward
        assert baseline_get_microbatch_id_in_model_chunk(
            i, forward=True
        ) == new_get_microbatch_id_in_model_chunk(i, forward=True)

        assert baseline_is_first_microbatch_for_model_chunk(
            i
        ) == new_is_first_microbatch_for_model_chunk(i)
        assert baseline_is_last_microbatch_for_model_chunk(
            i
        ) == new_is_last_microbatch_for_model_chunk(i)


def test_helpers():
    for pp in [2, 4, 8]:
        for m in [pp, 2 * pp, 4 * pp, 8 * pp]:
            for vp in range(2, 13):
                compare_helpers(pipeline_parallel_size=pp, num_microbatches=m, num_model_chunks=vp)
