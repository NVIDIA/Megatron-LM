
#include <algorithm>
#include <iostream>
#include <limits>
#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace std;


inline uint32_t get_sample_len(const int short_seq_ratio,
			       const uint32_t max_length) {
  /* Training sample length. */
  const auto random_number = rand();
  if ((random_number % short_seq_ratio) == 0) {
    return 2 + random_number % (max_length - 1);
  }
  return max_length;
}


py::array_t<uint32_t> build_mapping(const py::array_t<uint32_t>& docs_,
				    const py::array_t<uint16_t>& sizes_,
				    const int num_epochs,
				    const int max_num_samples,
				    const int max_seq_length,
				    const double short_seq_prob,
				    const int seed) {

  cout << "> building dataset mapping for " << docs_.shape(0) - 1 <<
    " documents with " << sizes_.shape(0) << " sentences ..." << endl;

  // For efficiency, convert probability to ratio.
  const int short_seq_ratio = int(round(1.0 / short_seq_prob));

  // Remove bound checks.
  auto docs = docs_.unchecked<1>();
  auto sizes = sizes_.unchecked<1>();

  // Check for consistency.
  if (docs[docs.shape(0) - 1] != sizes.shape(0)) {
    cout << "document values is not consistent with length of sizes: " <<
      docs[docs.shape(0) - 1] << " != " << sizes.shape(0) << endl;
    throw(-1);
    }

  // Mapping and it's length (1D).
  int num_samples = -1;
  uint32_t* maps = NULL;

  // Perform two iterations, in the first iteration get the size
  // and allocate memory and in the second iteration populate the map.
  bool second = false;
  for (int iteration=0; iteration < 2; ++iteration) {

    // Set the seed so both iterations produce the same results.
    srand(seed);

    // Set the flag on second iteration.
    if (iteration == 1) {
      second = true;
    }

    // Counters:
    uint32_t empty_docs = 0;
    uint32_t one_sent_docs = 0;

    // Current map index.
    uint64_t map_index = 0;

    // For each epoch:
    for (int epoch=0; epoch < num_epochs; ++epoch) {
      if (map_index >= max_num_samples) {
	cout << " > reached " << max_num_samples << " samples after " <<
	  epoch << " epochs ..." << endl;
	break;
      }
      // For each document:
      for (int doc=0; doc < (docs.shape(0) - 1); ++doc) {

	// Document sentences are in [sent_index_first, sent_index_last).
	const uint32_t sent_index_first = docs[doc];
	const uint32_t sent_index_last = docs[doc + 1];

	// At the begining of the document previous index is the start index.
	uint32_t prev_start_index = sent_index_first;

	// Remaining documents.
	uint32_t num_remain_sent = sent_index_last - sent_index_first;

	// Some bookkeeping
	if ((epoch == 0) && (!second)) {
	  if (num_remain_sent == 0) {
	    cout << "***WARNING*** document " << doc << " is empty" << endl;
	    empty_docs += 1;
	  }
	  if (num_remain_sent == 1) {
	    cout << "***WARNING*** document " << doc <<
	      " has one sentence" << endl;
	    one_sent_docs += 1;
	  }
	}

	// If we have more than two sentences.
	if (num_remain_sent > 1) {

	  // Set values.
	  uint32_t size = 0;
	  uint32_t num_sent = 0;
	  uint32_t seq_len = get_sample_len(short_seq_ratio, max_seq_length);

	  // Loop through sentences.
	  for (uint32_t sent_index=sent_index_first;
	       sent_index < sent_index_last; ++sent_index) {

	    // Add the size and number of sentences.
	    size += sizes[sent_index];
	    num_sent += 1;
	    num_remain_sent -= 1;

	    // If we have reached the target length.
	    // and if not only one sentence is left in the document.
	    // and if we have at least two sentneces.
	    // and if we have reached end of the document.
	    if (((size >= seq_len) && (num_remain_sent > 1) &&
		 (num_sent > 1) ) || (num_remain_sent == 0)) {

	      // Populate the map.
	      if (second) {
		const uint64_t map_index_0 = 3 * map_index;
		maps[map_index_0] = prev_start_index;
		maps[map_index_0 + 1] = sent_index + 1;
		maps[map_index_0 + 2] = seq_len;
	      }

	      // Update indices / counters.
	      map_index += 1;
	      prev_start_index = sent_index + 1;
	      seq_len = get_sample_len(short_seq_ratio, max_seq_length);
	      size = 0;
	      num_sent = 0;
	    }
	  }

	} // if (num_remain_sent > 1) {
      } // for (int doc=0; doc < num_docs; ++doc) {
    } // for (int epoch=0; epoch < num_epochs; ++epoch) {

    // For now only support mappings up to MAX_INT.
    if (map_index > std::numeric_limits<int>::max()) {
      cout << "number of samples ("<< map_index <<") exceeded MAX_INT" << endl;
      throw(-1);
    }
    else if (!second) {
      cout << "    number of samples:                      " <<
	map_index << endl;
      cout << "    number of empty documents:              " <<
	empty_docs << endl;
      cout << "    number of documents with one sentence:  " <<
	one_sent_docs << endl;
      maps = new uint32_t[3*map_index];
      num_samples = int(map_index);
    }

  } // for (int iteration=0; iteration < 2; ++iteration) {

  // Shuffle.
  for (int i=(num_samples - 1); i > 0; --i) {
    const int j = rand() % (i + 1);
    uint64_t i0 = 3 * i;
    uint64_t j0 = 3 * j;
    // Swap values.
    swap(maps[i0], maps[j0]);
    swap(maps[i0 + 1], maps[j0 + 1]);
    swap(maps[i0 + 2], maps[j0 + 2]);
  }

  cout << " > done building the mapping." << endl;

  // Method to deallocate memory.
  py::capsule free_when_done(maps, [](void *mem_) {
      uint32_t *mem = reinterpret_cast<uint32_t *>(mem_);
      cout << "freeing memory for the dataset mapping" << endl;
      delete[] mem;
    });

  // Return the numpy array.
  return py::array_t<uint32_t>({num_samples, 3}, // shape
			       {3*4, 4}, // C-style contiguous strides
			       maps, // the data pointer
			       free_when_done); // numpy array references

}


PYBIND11_MODULE(helpers, m) {
  m.def("build_mapping", &build_mapping);
}


