#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cif_dict_pybind.h"
#include "fasta_iterator_pybind.h"
#include "msa_conversion_pybind.h"

PYBIND11_MODULE(parsers_cpp, m)
{
    m.doc() = R"pbdoc(
        DNA-Flex C++ accelerated parsers module
        -------------------------------------
        .. currentmodule:: parsers_cpp
    )pbdoc";
    
    dnaflex::parsers::init_cif_dict(m);
    dnaflex::parsers::init_fasta_iterator(m);
    dnaflex::parsers::init_msa_conversion(m);
}