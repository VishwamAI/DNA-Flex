#include "msa_profile_pybind.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dnaflex {
namespace parsers {

void init_msa_profile(py::module_& m) {
    py::class_<MSAProfile>(m, "MSAProfile")
        .def(py::init<>())
        .def("add_sequence", &MSAProfile::add_sequence)
        .def("compute_profile", &MSAProfile::compute_profile)
        .def("get_position_weights", &MSAProfile::get_position_weights)
        .def("get_conservation_scores", &MSAProfile::get_conservation_scores)
        .def("get_sequence_identity", &MSAProfile::get_sequence_identity)
        .def("get_gap_frequencies", &MSAProfile::get_gap_frequencies)
        .def("get_nucleotide_frequencies", &MSAProfile::get_nucleotide_frequencies)
        .def("get_consensus_sequence", &MSAProfile::get_consensus_sequence)
        .def("get_variable_regions", &MSAProfile::get_variable_regions)
        .def("get_alignment_length", &MSAProfile::get_alignment_length)
        .def("get_sequence_count", &MSAProfile::get_sequence_count)
        .def("get_average_sequence_length", &MSAProfile::get_average_sequence_length);
}

} // namespace parsers
} // namespace dnaflex