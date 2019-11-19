
#include <algorithm>
#include <iostream>
#include <limits>
#include <math.h>
#include <stdexcept>
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

template<typename DocIdx>
py::array build_mapping_impl(const py::array_t<uint32_t>& docs_,
                             const py::array_t<uint16_t>& sizes_,
                             const int num_epochs,
                             const uint64_t max_num_samples,
                             const int max_seq_length,
                             const double short_seq_prob,
                             const int seed) {

    cout << "> building dataset mapping for " << docs_.shape(0) - 1\
         << " documents with " << sizes_.shape(0) << " sentences ..."
         << std::flush << endl;

    // For efficiency, convert probability to ratio.
    const auto short_seq_ratio = static_cast<int>(round(1.0 / short_seq_prob));

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
    int64_t num_samples = -1;
    DocIdx* maps = NULL;

    // Perform two iterations, in the first iteration get the size
    // and allocate memory and in the second iteration populate the map.
    bool second = false;
    for (int iteration=0; iteration < 2; ++iteration) {

        // Set the seed so both iterations produce the same results.
        srand(seed);

        // Set the flag on second iteration.
        second = iteration == 1;

        // Counters:
        uint32_t empty_docs = 0;
        uint32_t one_sent_docs = 0;

        // Current map index.
        uint64_t map_index = 0;

        // For each epoch:
        for (int epoch=0; epoch < num_epochs; ++epoch) {
            if (map_index >= max_num_samples && !second) {
                cout << " > reached " << max_num_samples << " samples after "
                     << epoch << " epochs ..." << std::flush << endl;
                break;
            }
            // For each document:
            for (int doc=0; doc < (docs.shape(0) - 1); ++doc) {

                // Document sentences are in [sent_index_first, sent_index_last).
                const auto sent_index_first = docs[doc];
                const auto sent_index_last = docs[doc + 1];

                // At the begining of the document previous index is the start index.
                auto prev_start_index = sent_index_first;

                // Remaining documents.
                auto num_remain_sent = sent_index_last - sent_index_first;

                // Some bookkeeping
                if ((epoch == 0) && (!second)) {
                    if (num_remain_sent == 0) {
                        cout << "***WARNING*** document " << doc << " is empty" << endl;
                        empty_docs += 1;
                    }
                    if (num_remain_sent == 1) {
                        // cout << "***WARNING*** document " << doc <<
                        //         " has one sentence" << endl;
                        one_sent_docs += 1;
                    }
                }

                // If we have more than two sentences.
                if (num_remain_sent > 1) {

                    // Set values.
                    auto size = uint32_t{0};
                    auto num_sent = uint32_t{0};
                    auto seq_len = get_sample_len(short_seq_ratio, max_seq_length);

                    // Loop through sentences.
                    for (auto sent_index=sent_index_first;
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
                                const auto map_index_0 = 3 * map_index;
                                maps[map_index_0] = prev_start_index;
                                maps[map_index_0 + 1] = sent_index + 1;
                                maps[map_index_0 + 2] = seq_len;
                            }

                            // Update indices / counters.
                            // check for overflow
                            if (map_index == std::numeric_limits<DocIdx>::max()) {
                                cout << "number of samples exceeded maximum allowed by type: "
                                     << std::numeric_limits<DocIdx>::max() << endl;
                                throw std::overflow_error("Number of samples");
                            }
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

        if (!second) {
            cout << "    number of samples:                      " <<
                    map_index << endl;
            cout << "    number of empty documents:              " <<
                    empty_docs << endl;
            cout << "    number of documents with one sentence:  " <<
                    one_sent_docs << endl;
            maps = new DocIdx[3*map_index];
            num_samples = map_index;
        }

    } // for (int iteration=0; iteration < 2; ++iteration) {

    // Shuffle.
    for (auto i=(num_samples - 1); i > 0; --i) {
        const auto j = rand() % (i + 1);
        const auto i0 = 3 * i;
        const auto j0 = 3 * j;
        // Swap values.
        swap(maps[i0], maps[j0]);
        swap(maps[i0 + 1], maps[j0 + 1]);
        swap(maps[i0 + 2], maps[j0 + 2]);
    }

    cout << " > done building the mapping." << endl;

    // Method to deallocate memory.
    py::capsule free_when_done(maps, [](void *mem_) {
            DocIdx *mem = reinterpret_cast<DocIdx*>(mem_);
            cout << "freeing memory for the dataset mapping" << endl;
            delete[] mem;
        });

    // Return the numpy array.
    return py::array(std::vector<int64_t>{num_samples, 3}, // shape
                     {3*4, 4}, // C-style contiguous strides
                     maps, // the data pointer
                     free_when_done); // numpy array references

}

py::array build_mapping(const py::array& docs_,
                        const py::array& sizes_,
                        const int num_epochs,
                        const uint64_t max_num_samples,
                        const int max_seq_length,
                        const double short_seq_prob,
                        const int seed) {
    if (sizes_.size() > std::numeric_limits<uint32_t>::max()) {
        return build_mapping_impl<uint64_t>(docs_, sizes_, num_epochs, max_num_samples,
                                            max_seq_length, short_seq_prob, seed);
    } else {
        return build_mapping_impl<uint32_t>(docs_, sizes_, num_epochs, max_num_samples,
                                            max_seq_length, short_seq_prob, seed);
    }
}

PYBIND11_MODULE(helpers, m) {
    m.def("build_mapping", &build_mapping);
}
