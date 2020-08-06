//
// Created by sergio on 16/05/19.
//

#include "../../include/Model.h"
#include "../../include/Tensor.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>



#include <numeric>
#include <iomanip>

namespace py = pybind11;

//class Singleton {
//   static Singleton *instance;
//   Model model;
//   Tensor input_a;
//   Tensor output;
//
//   // Private constructor so that no objects can be created.
//   Singleton() {
//    Model model("mobilenet_v2_1.4_224_frozen.pb");
//    Tensor input_a{model, "input"};
//    Tensor output{model, "MobilenetV2/Predictions/Reshape_1"};
//   }
//
//   public:
//   static Singleton *getInstance() {
//      if (!instance)
//      instance = new Singleton;
//      return instance;
//   }
//
//   Model getModel() {
//      return this -> model;
//   }
//
//   Tensor getInput() {
//      return this -> input_a;
//   }
//
//   Tensor getOutput() {
//      return this -> output;
//   }
//
//};


Model model("/model/mobilenet_v2_1.4_224_frozen.pb");
Tensor input_a{model, "input"};
Tensor output{model, "MobilenetV2/Predictions/Reshape_1"};



int run(std::vector<float>  input1, std::vector<float>  input2) {
    // Load model with a path to the .pb file. 
    // An optional std::vector<uint8_t> parameter can be used to supply Tensorflow with
    // session options. The vector must represent a serialized ConfigProto which can be 
    // generated manually in python. See create_config_options.py.
    // Example:
    // const std::vector<uint8_t> ModelConfigOptions = { 0x32, 0xb, 0x9, 0x9a, 0x99, 0x99, 0x99, 0x99, 0x99, 0xb9, 0x3f, 0x20, 0x1 };
    // Model model("../model.pb", ModelConfigOptions);


    Model model("model.pb");
    model.init();

    Tensor input_a{model, "input_a"};
    Tensor input_b{model, "input_b"};
    Tensor output{model, "result"};

    //std::vector<float> data(100);


    //std::iota(data.begin(), data.end(), 0);

    input_a.set_data(input1);
    input_b.set_data(input2);

    model.run({&input_a, &input_b}, output);
    for (float f : output.get_data<float>()) {
        std::cout << f << " ";
    }
    std::cout << std::endl;

}


std::vector<float> run1(std::vector<float>  input1) {
    // Load model with a path to the .pb file.
    // An optional std::vector<uint8_t> parameter can be used to supply Tensorflow with
    // session options. The vector must represent a serialized ConfigProto which can be
    // generated manually in python. See create_config_options.py.
    // Example:
    // const std::vector<uint8_t> ModelConfigOptions = { 0x32, 0xb, 0x9, 0x9a, 0x99, 0x99, 0x99, 0x99, 0x99, 0xb9, 0x3f, 0x20, 0x1 };
    // Model model("../model.pb", ModelConfigOptions);


    //Model model("mobilenet_v2_1.4_224_frozen.pb");

//    for (std::string f : model.get_operations()) {
//        std::cout << f << " ";
//    }
//
//    model.init();

    //Tensor input_a{model, "input"};
    //Tensor output{model, "MobilenetV2/Predictions/Reshape_1"};

    //std::vector<float> data(100);


    //std::iota(data.begin(), data.end(), 0);

//    Singleton *s = s->getInstance();
//    Model model=s.getModel();
//    Tensor input_a=s.getInput();
//    Tensor output=s.getOutput();

//    std::string file_path = __FILE__;
//    std::string dir_path = file_path.substr(0, file_path.rfind("/"));
//    std::cout<<dir_path<<std::endl;

    std::cout << "running model" <<std::endl;
    input_a.set_data(input1, {3, 224, 224, 3});

    model.run({&input_a}, output);
    std::vector<float> res = output.get_data<float>();
//    for (float f : res) {
//        std::cout << f << " ";
//    }
//    std::cout << std::endl;

    return res;

}


int add(int i, int j) {
    return i + j;
}


py::array_t<double> add_arrays(py::array_t<double> input1, py::array_t<double> input2) {
    py::buffer_info buf1 = input1.request(), buf2 = input2.request();

    if (buf1.ndim != 1 || buf2.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");

    if (buf1.size != buf2.size)
        throw std::runtime_error("Input shapes must match");

    /* No pointer is passed, so NumPy will allocate the buffer */
    auto result = py::array_t<double>(buf1.size);
    py::buffer_info buf3 = result.request();

    double *ptr1 = (double *) buf1.ptr,
           *ptr2 = (double *) buf2.ptr,
           *ptr3 = (double *) buf3.ptr;

    for (size_t idx = 0; idx < buf1.shape[0]; idx++)
        ptr3[idx] = ptr1[idx] + ptr2[idx];

    return result;
}




PYBIND11_MODULE(cmake_example, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("run", &run, R"pbdoc(
        Run inference

        Some other explanation about the run function.
    )pbdoc");

    m.def("add_arrays", &add_arrays, "Add two NumPy arrays");
    m.def("run1", &run1, "Run inference");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}



