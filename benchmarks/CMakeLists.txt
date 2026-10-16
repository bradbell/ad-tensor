# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
#[[
{xrst_begin benchmarks ben}
{xrst_spell
    autograd
    podman
    dockerfile
    rmi
}

Comparing Execution Time for AD Tensor Versus Autograd
######################################################

Test Problems
*************
{xrst_toc_table
    benchmarks/fit_poly.cpp
    benchmarks/multi_normal.cpp
    benchmarks/smooth_spline.cpp
}

Tutorial
********
The benchmarks above also serve as a tutorials for using AD Tensors.

Running Benchmarks on Your Machine
**********************************
If podman is installed and the Dockerfile below
is in the current working directory,
The following commands (in a bash shell) will run the benchmarks:
{xrst_code sh}
    # create the ad-tensor image
    podman build -f Dockerfile -t ad-tensor
    #
    # create and run a container starting with the image
    podman run -it ad-tensor bash
    #
    # run the benchmarks and generate the summary report
    tools/run_benchmarks.sh
    #
    # exit the container
    exit
    #
    # remove the image and container
    podman rmi --force ad-tensor
{xrst_code}

Dockerfile
**********
{xrst_spell_off}
{xrst_code dockerfile}
# initialize system
FROM ubuntu:latest
RUN  apt update
#
# install packages required by ad-tensor
RUN apt install git cmake ninja-build  libgtest-dev -y
RUN apt install libboost-system-dev libboost-filesystem-dev -y
RUN apt install python3 python3-pip python3-venv -y
#
# install pytorch in a virtual environment
RUN python3 -m venv my_env
ENV PATH="$PWD/my_env/bin:$PATH"
RUN pip3 install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu
#
# ad-tensor.git
RUN     git clone https://github.com/bradbell/ad-tensor.git ad-tensor.git
WORKDIR ad-tensor.git
#
# Build and test examples, tests, and benchmarks
RUN tools/check_gtest.sh --release
{xrst_code}
{xrst_spell_on}

{xrst_end benchmarks}
-------------------------------------------------------------------------------
]]
#
# BEGIN_SORT_THIS_LINE_PLUS_2
add_executable(benchmarks
    fit_poly.cpp
    multi_normal.cpp
    smooth_spline.cpp
)
# END_SORT_THIS_LINE_MINUS_2
#
target_link_libraries(benchmarks PRIVATE
    ad_tensor
    GTest::Main
    ${TORCH_LIBRARIES}
    ${plugin_LIBRARIES}
)
