from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='custom_matmul_cpp',
    ext_modules=[
        CppExtension(
            name='custom_matmul_cpp',
            sources=[
                'matmul.cpp',
                'pytorch_wrapper.cpp'
            ],
            include_dirs=['./']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)