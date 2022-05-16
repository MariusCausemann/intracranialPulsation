from fenics import *

meshExpression_code = """

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/mesh/MeshFunction.h>

class meshExpression : public dolfin::Expression
{
public:

  // Create expression with 1 components
  meshExpression() : dolfin::Expression() {}

  // Function for evaluating expression on each cell
  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const override
  {
    const uint cell_index = cell.index;
    values[0] = (*dom)[cell_index];
  }

  // The data stored in mesh functions
  std::shared_ptr<dolfin::MeshFunction<size_t>> dom;

};

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<meshExpression, std::shared_ptr<meshExpression>, dolfin::Expression>
    (m, "meshExpression")
    .def(py::init<>())
    .def_readwrite("dom", &meshExpression::dom);
}

"""

meshExpression = compile_cpp_code(meshExpression_code).meshExpression()