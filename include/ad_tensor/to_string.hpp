#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <string>
#include <torch/torch.h>
#include <ad_tensor/ad_type.hpp>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/op_enum.hpp>
#include <ad_tensor/vector.hpp>
namespace ad_tensor {
    std::string to_string(const vector<at::Tensor>& vec);
    std::string to_string(const vector<adten_t>& avec);
}
