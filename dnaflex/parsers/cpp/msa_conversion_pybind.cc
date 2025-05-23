#include "msa_conversion_pybind.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace dnaflex {
namespace parsers {

void init_msa_conversion(py::module_& m) {
    py::class_<MSAConverter>(m, "MSAConverter")
        .def(py::init<>())
        .def("fasta_to_matrix", &MSAConverter::fasta_to_matrix)
        .def("matrix_to_fasta", &MSAConverter::matrix_to_fasta)
        .def("validate_alignment", &MSAConverter::validate_alignment)
        .def("get_alignment_length", &MSAConverter::get_alignment_length)
        .def("get_sequence_count", &MSAConverter::get_sequence_count);
}

} // namespace parsers
} // namespace dnaflex