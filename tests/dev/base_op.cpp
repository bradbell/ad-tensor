// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <torch/torch.h>
//
#include <ad_tensor/vector.hpp>
#include <ad_tensor/ad_type.hpp>
#include <ad_tensor/dev/op_enum.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/derive_op.hpp>
//
// empty namespace
namespace {
    //
    // op_enum_t, ad_type_t
    using ad_tensor::dev::op_enum_t;
    using ad_tensor::ad_type_t;
    using ad_tensor::dev::agraph_t;
    using ad_tensor::dev::base_op_t;
    //
    // vector
    using ad_tensor::vector;
    //
    // Tensor
    using at::Tensor;
}

TEST(tests_dev, op_base)  {
    //
    // base_op
    const base_op_t<at::Tensor>& base_op =
        ad_tensor::dev::op_enum2derive_op(op_enum_t::add);
    //
    // op_enum
    EXPECT_EQ( op_enum_t::add, base_op.op_enum() );
    //
    // m_arg_start, m_arg_value, m_arg_type, con_vec, par_vec
    op_enum_t              dom         = op_enum_t::dom;
    op_enum_t              add         = op_enum_t::add;
    ad_type_t              par         = ad_type_t::parameter;
    Tensor                 empty       = torch::empty( {0} );
    Tensor                 ones        = torch::ones( {2} );
    size_t                 op_index    = 1;
    ad_tensor::vector<Tensor>    con_vec     = {};
    ad_tensor::vector<Tensor>    par_vec     = {ones, empty};
    agraph_t               agraph;
    agraph.m_op_seq     = ad_tensor::vector<op_enum_t>( {dom, add} );
    agraph.m_arg_start  = ad_tensor::vector<size_t>( {0, 0, 2} );
    agraph.m_arg_value  = ad_tensor::vector<size_t>( {0, 0} );
    agraph.m_arg_type   = ad_tensor::vector<ad_type_t>( {par, par} );
    //
    // par_vec[op_index]
    base_op.forward_par(op_index, agraph, con_vec, par_vec);
    //
    // data_ptr
    float* data_ptr = par_vec[op_index].data_ptr<float>();
    EXPECT_EQ( data_ptr[0], 2.0 );
    EXPECT_EQ( data_ptr[1], 2.0 );
}
