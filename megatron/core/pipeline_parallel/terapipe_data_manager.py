import torch
import math
from megatron import get_args


class TeraPipeDataManager:
    '''
    Wrap data_iterator, slice data's sequence dim into slice.  
    Length of slice is mbs_seq_length + 1.
    If data_iterator is None, then return None.
    '''
    def __init__(self, data_iterator, seq_slice_length, batch_dim=0, sequence_dim=1):
        self.args = get_args()
        self.data_iterator = data_iterator
        self.seq_slice_length = seq_slice_length
        self.batch_dim = batch_dim
        self.sequence_dim = sequence_dim
        self.current_slice = 0  # Initialize slice index
        self.slice_data()

    def slice_data(self):
        data = next(self.data_iterator)
        data_seq_len = data['text'].size(self.sequence_dim)
        data_batch_size = data['text'].size(self.batch_dim)
        assert data_seq_len >= self.seq_slice_length
        # data would have 1 more sequence in order to get label for the last token
        assert (data_seq_len-1)%self.seq_slice_length == 0, 'seq_length should be divisible by terapipe_slice_len'
        window_size = self.seq_slice_length + 1 
        # Calculate number of complete slices
        num_complete_slices = data_seq_len // window_size
        # Check if there's a remainder to determine if an extra slice is needed
        remainder = data_seq_len % window_size
        self.slices = []
        for i in range(num_complete_slices):
            start_idx = i * self.seq_slice_length
            end_idx = start_idx + window_size
            self.slices.append(data['text'][:, start_idx:end_idx])

        if remainder > 0:
            # For the last slice, start index is adjusted to ensure the slice has seq_slice_length + 1 length
            last_start_idx = data_seq_len - window_size
            self.slices.append(data['text'][:, last_start_idx:])

        self.num_slices = len(self.slices)

    def __next__(self):
        if self.current_slice >= self.num_slices:
            self.slice_data()  # Get new data and reset current_slice
            self.current_slice = 0
        slice_ = self.slices[self.current_slice]
        self.current_slice += 1
        data = {}
        data['text'] = slice_
        return data


    def __iter__(self):
        return self

    def __len__(self):
        return self.num_slices