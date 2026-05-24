// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <torch/torch.h>
//
#include <ad_tensor/ad_type.hpp>
#include <ad_tensor/dev/op_enum.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/op_derive.hpp>
//
// empty namespace
namespace {
    //
    // op_enum_t, ad_type_t
    using ad_tensor::dev::op_enum_t;
    using ad_tensor::ad_type_t;
    using ad_tensor::dev::agraph_t;
    using ad_tensor::dev::op_base_t;
    //
    // vector
    using std::vector;
    //
    // Tensor
    using at::Tensor;
}

TEST(tests_devel, op_base)  {
    //
    // add_op
    ad_tensor::dev::add_op_t add_op;
    //
    // op_enum
    EXPECT_EQ( op_enum_t::add, add_op.op_enum() );
    //
    // trace, m_arg_start, m_arg_value, m_arg_type, con_vec, par_vec
    op_enum_t              dom         = op_enum_t::dom;
    op_enum_t              add         = op_enum_t::add;
    ad_type_t              par         = ad_type_t::parameter;
    Tensor                 empty       = torch::empty( {0} );
    Tensor                 ones        = torch::ones( {2} );
    bool                   trace       = false;
    size_t                 op_index    = 1;
    std::vector<Tensor>    con_vec     = {};
    std::vector<Tensor>    par_vec     = {ones, empty};
    agraph_t               agraph;
    agraph.m_op_seq     = std::vector<op_enum_t>( {dom, add} );
    agraph.m_arg_start  = std::vector<size_t>( {0, 0, 2} );
    agraph.m_arg_value  = std::vector<size_t>( {0, 0} );
    agraph.m_arg_type   = std::vector<ad_type_t>( {par, par} );
    //
    // par_vec[op_index]
    add_op.forward_par(op_index, agraph, con_vec, par_vec);
    //
    // data_ptr
    float* data_ptr = par_vec.at(op_index).data_ptr<float>();
    EXPECT_EQ( data_ptr[0], 2.0 );
    EXPECT_EQ( data_ptr[1], 2.0 );
}
