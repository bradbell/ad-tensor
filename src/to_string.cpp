// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/to_string.hpp>
#include <ad_tensor/to_string.hpp>
//
namespace ad_tensor {
    std::string to_string(const vector<at::Tensor>& vec) {
        return dev::to_string(vec);
    }
    std::string to_string(const vector<adten_t>& vec) {
        return dev::to_string(vec);
    }
}
