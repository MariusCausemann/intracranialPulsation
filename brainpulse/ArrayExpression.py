from dolfin import *

code = """
    #include <pybind11/eigen.h>
    #include <dolfin/function/Expression.h>
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    namespace py = pybind11;
    
    typedef Eigen::VectorXd npArray;

    class ArrayExpression : public dolfin::Expression {
      public:
        npArray arr;
        int i = 0;
        double t = 0.0;
        
        ArrayExpression(npArray a) : dolfin::Expression(), arr(a) {}

        void eval(Eigen::Ref<Eigen::VectorXd> values, 
                  Eigen::Ref<const Eigen::VectorXd> x) const {
            
            values[0] = arr[i];
        }
    };
    
    PYBIND11_MODULE(SIGNATURE, m) {
        pybind11::class_<ArrayExpression, std::shared_ptr<ArrayExpression>, 
                         dolfin::Expression>
        (m, "ArrayExpression")
        .def(pybind11::init<npArray>())
        .def_readwrite("i", &ArrayExpression::i)
        .def_readwrite("t", &ArrayExpression::t)
        ;
    }
    
    """

def getArrayExpression(array, degree=1):
    return CompiledExpression(compile_cpp_code(code).ArrayExpression(array), degree=degree)


class InterpolatedSource(UserExpression):
    def __init__(self, path, delimiter=",", t=0, **kwargs):
        data = np.loadtxt(path)
        times = data[:,0]
        inflow = data[:,1]
        self.t = 0.0
        self.interp = interpolate.interp1d(times, inflow)
        super().__init__(**kwargs)
    def eval(self, values, x):
        return self.interp(self.t)