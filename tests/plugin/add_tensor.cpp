// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <boost/config.hpp>
#include <boost/dll/alias.hpp>
#include <iostream>
#include <torch/torch.h>

// add_tensor_plugin
at::Tensor add_tensor_plugin(const at::Tensor& x, const at::Tensor& y) {
    return x + y;
}

// add_tensor
BOOST_DLL_ALIAS(add_tensor_plugin, add_tensor)
