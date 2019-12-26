/*
 coding=utf-8
 Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */


/* Helper methods for fast index mapping builds */

#include <algorithm>
#include <iostream>
#include <limits>
#include <math.h>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>

namespace py = pybind11;
using namespace std;

const int32_t LONG_SENTENCE_LEN = 512;


inline int32_t get_target_sample_len(const int32_t short_seq_ratio,
				     const int32_t max_length,
				     std::mt19937& rand32_gen) {
    /* Training sample length. */
    const auto random_number = rand32_gen();
    if ((random_number % short_seq_ratio) == 0) {
      return 2 + random_number % (max_length - 1);
    }
    return max_length;
}


template<typename DocIdx>
py::array build_mapping_impl(const py::array_t<int64_t>& docs_,
                             const py::array_t<int32_t>& sizes_,
                             const int32_t num_epochs,
                             const uint64_t max_num_samples,
                             const int32_t max_seq_length,
                             const double short_seq_prob,
                             const int32_t seed,
			     const bool verbose) {
    /* Build a mapping of (start-index, end-index, sequence-length) where
       start and end index are the indices of the sentences in the sample
       and sequence-length is the target sequence length.
    */

    // Consistency checks.
    assert(num_epochs > 0);
    assert(max_seq_length > 1);
    assert(short_seq_prob > 0.0);
    assert(short_seq_prob <= 1.0);
    assert(seed > 0);

    // Remove bound checks.
    auto docs = docs_.unchecked<1>();
    auto sizes = sizes_.unchecked<1>();

    // For efficiency, convert probability to ratio. Note: rand() generates int.
    const auto short_seq_ratio = static_cast<int32_t>(round(1.0 / short_seq_prob));

    if (verbose) {
        const auto sent_start_index = docs[0];
	const auto sent_end_index = docs[docs_.shape(0) - 1];
	const auto num_sentences = sent_end_index - sent_start_index;
	cout << "    using:" << endl << std::flush;
	cout << "     number of documents:            " << docs_.shape(0) - 1 <<
	  endl << std::flush;
	cout << "     sentences range:                [" << sent_start_index <<
	", " << sent_end_index << ")" << endl << std::flush;
	cout << "     total number of sentences:      " << num_sentences <<
	  endl << std::flush;
	cout << "     number of epochs:               " << num_epochs <<
	  endl << std::flush;
	cout << "     maximum number of samples:      " << max_num_samples <<
	  endl << std::flush;
	cout << "     maximum sequence length:        " << max_seq_length <<
	  endl << std::flush;
	cout << "     short sequence probability:     " << short_seq_prob <<
	endl << std::flush;
	cout << "     short sequence ration (1/prob): " << short_seq_ratio <<
	  endl << std::flush;
	cout << "     seed:                           " << seed << endl <<
	  std::flush;
    }

    // Mapping and it's length (1D).
    int64_t num_samples = -1;
    DocIdx* maps = NULL;

    // Perform two iterations, in the first iteration get the size
    // and allocate memory and in the second iteration populate the map.
    bool second = false;
    for (int32_t iteration=0; iteration<2; ++iteration) {

        // Set the seed so both iterations produce the same results.
        std::mt19937 rand32_gen(seed);

        // Set the flag on second iteration.
        second = (iteration == 1);

        // Counters:
        uint64_t empty_docs = 0;
        uint64_t one_sent_docs = 0;
	uint64_t long_sent_docs = 0;

        // Current map index.
        uint64_t map_index = 0;

        // For each epoch:
        for (int32_t epoch=0; epoch<num_epochs; ++epoch) {
            if (map_index >= max_num_samples) {
	        if (verbose && (!second)) {
		  cout << "    reached " << max_num_samples << " samples after "
		       << epoch << " epochs ..." << endl << std::flush;
		}
                break;
            }
            // For each document:
            for (int32_t doc=0; doc<(docs.shape(0) - 1); ++doc) {

                // Document sentences are in [sent_index_first, sent_index_last)
                const auto sent_index_first = docs[doc];
                const auto sent_index_last = docs[doc + 1];

                // At the begining of the document previous index is the
		// start index.
                auto prev_start_index = sent_index_first;

                // Remaining documents.
                auto num_remain_sent = sent_index_last - sent_index_first;

                // Some bookkeeping
                if ((epoch == 0) && (!second)) {
                    if (num_remain_sent == 0) {
		        ++empty_docs;
                    }
                    if (num_remain_sent == 1) {
		        ++one_sent_docs;
                    }
                }

		// Detect documents with long sentences.
		bool contains_long_sentence = false;
		if (num_remain_sent > 1) {
		    for (auto sent_index=sent_index_first;
			 sent_index < sent_index_last; ++sent_index) {
		        if (sizes[sent_index] > LONG_SENTENCE_LEN){
			    if ((epoch == 0) && (!second)) {
			        ++long_sent_docs;
			    }
			    contains_long_sentence = true;
			    break;
			}
		    }
		}

                // If we have more than two sentences.
                if ((num_remain_sent > 1) && (!contains_long_sentence)) {

                    // Set values.
                    auto seq_len = int32_t{0};
                    auto num_sent = int32_t{0};
                    auto target_seq_len = get_target_sample_len(short_seq_ratio,
								max_seq_length,
								rand32_gen);

                    // Loop through sentences.
                    for (auto sent_index=sent_index_first;
                         sent_index < sent_index_last; ++sent_index) {

		        // Add the size and number of sentences.
		        seq_len += sizes[sent_index];
		        ++num_sent;
			--num_remain_sent;

			// If we have reached the target length.
			// and if not only one sentence is left in the document.
			// and if we have at least two sentneces.
			// and if we have reached end of the document.
			if (((seq_len >= target_seq_len) &&
			     (num_remain_sent > 1) &&
			     (num_sent > 1) ) || (num_remain_sent == 0)) {

			    // Check for overflow.
			    if ((3 * map_index + 2) >
				std::numeric_limits<int64_t>::max()) {
			        cout << "number of samples exceeded maximum "
				     << "allowed by type int64: "
				     << std::numeric_limits<int64_t>::max()
				     << endl;
				throw std::overflow_error("Number of samples");
			    }

			    // Populate the map.
			    if (second) {
			        const auto map_index_0 = 3 * map_index;
				maps[map_index_0] = static_cast<DocIdx>(prev_start_index);
				maps[map_index_0 + 1] = static_cast<DocIdx>(sent_index + 1);
				maps[map_index_0 + 2] = static_cast<DocIdx>(target_seq_len);
			    }

			    // Update indices / counters.
			    ++map_index;
			    prev_start_index = sent_index + 1;
			    target_seq_len = get_target_sample_len(short_seq_ratio,
								   max_seq_length,
								   rand32_gen);
			    seq_len = 0;
			    num_sent = 0;
			}

                    } // for (auto sent_index=sent_index_first; ...
                } // if (num_remain_sent > 1) {
            } // for (int doc=0; doc < num_docs; ++doc) {
        } // for (int epoch=0; epoch < num_epochs; ++epoch) {

        if (!second) {
	    if (verbose) {
	        cout << "   number of empty documents: " << empty_docs <<
		  endl << std::flush;
		cout << "   number of documents with one sentence: " <<
		  one_sent_docs << endl << std::flush;
		cout << "   number of documents with long sentences: " <<
		  long_sent_docs << endl << std::flush;
		cout << "   will create mapping for " << map_index <<
		  " samples" << endl << std::flush;
	    }
	    assert(maps == NULL);
	    assert(num_samples < 0);
            maps = new DocIdx[3*map_index];
            num_samples = static_cast<int64_t>(map_index);
        }

    } // for (int iteration=0; iteration < 2; ++iteration) {

    // Shuffle.
    // We need a 64 bit random number generator as we might have more
    // than 2 billion samples.
    std::mt19937_64 rand64_gen(seed + 1);
    for (auto i=(num_samples - 1); i > 0; --i) {
      const auto j = static_cast<int64_t>(rand64_gen() % (i + 1));
      const auto i0 = 3 * i;
      const auto j0 = 3 * j;
      // Swap values.
      swap(maps[i0], maps[j0]);
      swap(maps[i0 + 1], maps[j0 + 1]);
      swap(maps[i0 + 2], maps[j0 + 2]);
    }

    // Method to deallocate memory.
    py::capsule free_when_done(maps, [](void *mem_) {
            DocIdx *mem = reinterpret_cast<DocIdx*>(mem_);
	    delete[] mem;
        });

    // Return the numpy array.
    const auto byte_size = sizeof(DocIdx);
    return py::array(std::vector<int64_t>{num_samples, 3}, // shape
                     {3*byte_size, byte_size}, // C-style contiguous strides
                     maps, // the data pointer
                     free_when_done); // numpy array references

}


py::array build_mapping(const py::array_t<int64_t>& docs_,
                        const py::array_t<int>& sizes_,
                        const int num_epochs,
                        const uint64_t max_num_samples,
                        const int max_seq_length,
                        const double short_seq_prob,
                        const int seed,
			const bool verbose) {

    if (sizes_.size() > std::numeric_limits<uint32_t>::max()) {
        if (verbose) {
	   cout << "    using uint64 for data mapping..." << endl << std::flush;
	}
	return build_mapping_impl<uint64_t>(docs_, sizes_, num_epochs,
					    max_num_samples, max_seq_length,
					    short_seq_prob, seed, verbose);
    } else {
       if (verbose) {
	   cout << "    using uint32 for data mapping..." << endl << std::flush;
       }
       return build_mapping_impl<uint32_t>(docs_, sizes_, num_epochs,
					   max_num_samples, max_seq_length,
					   short_seq_prob, seed, verbose);
    }
}


PYBIND11_MODULE(helpers, m) {
    m.def("build_mapping", &build_mapping);
}
