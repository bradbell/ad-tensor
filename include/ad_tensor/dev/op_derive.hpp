#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/op_base.hpp>
//
#define OP_DERIVE(op_name) \
    struct op_name ## _t : public op_base_t { \
    \
    op_enum_t op_enum(void) const override; \
    \
    void forward_par( \
        size_t                            op_index    , \
        const agraph_t&                   agraph      , \
        const std::vector<at::Tensor>&    con_vec     , \
        std::vector<at::Tensor>&          par_vec     \
    ) const override; \
};
//
namespace ad_tensor { namespace dev {
    OP_DERIVE(add_op)
} }
