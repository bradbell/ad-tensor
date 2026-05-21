// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <torch/torch.h>
//
#include <ad_tensor/ad_type.hpp>
#include <ad_tensor/devel/op_enum.hpp>
#include <ad_tensor/devel/agraph.hpp>
#include <ad_tensor/devel/op_base.hpp>
//
// empty namespace
namespace {
    //
    // op_enum_t, ad_type_t
    using ad_tensor::devel::op_enum_t;
    using ad_tensor::ad_type_t;
    using ad_tensor::devel::agraph_t;
    using ad_tensor::devel::op_base_t;
    //
    // vector
    using std::vector;
    //
    // Tensor
    using at::Tensor;
    //
    // add_op_t
    class add_op_t : public  op_base_t {
    public:
        //
        // op_enum
        op_enum_t op_enum(void) const override {
            return op_enum_t::add;
        }
        //
        // forward_par
        void forward_par(
            bool                       trace       ,
            size_t                     op_index    ,
            const agraph_t&            agraph      ,
            const std::vector<Tensor>& con_vec     ,
            std::vector<Tensor>&       par_vec     ) const override
        {
            // op_index
            EXPECT_LT( 0, op_index );
            //
            // op_enum
            op_enum_t op_enum = agraph.op_vec.at(op_index);
            EXPECT_EQ( op_enum, op_enum_t::add );
            //
            // arg_start, arg_all, ad_type_all
            const std::vector<size_t>& arg_start        = agraph.arg_start;
            const std::vector<size_t>& arg_all          = agraph.arg_all;
            const std::vector<ad_type_t>& ad_type_all   = agraph.ad_type_all;
            //
            // n_arg
            size_t n_arg = arg_start.at(op_index+1) - arg_start.at(op_index);
            EXPECT_EQ( n_arg, 2 );
            //
            // arg_index
            size_t arg_first = arg_start.at(op_index);
            //
            // lhs_tensor
            size_t        lhs_index   = arg_all.at(arg_first);
            ad_type_t     lhs_ad_type = ad_type_all.at(arg_first);
            Tensor        lhs_tensor  = tensor_at_index(
               lhs_ad_type, lhs_index, con_vec, par_vec
            );
            //
            // rhs_tensor
            size_t        rhs_index   = arg_all.at(arg_first + 1);
            ad_type_t     rhs_ad_type = ad_type_all.at(arg_first + 1);
            Tensor        rhs_tensor  = tensor_at_index(
               rhs_ad_type, rhs_index, con_vec, par_vec
            );
            //
            // par_vec
            par_vec.at(op_index) = lhs_tensor + rhs_tensor;
        }
    };
}

TEST(tests_devel, op_base)  {
    //
    // add_op
    add_op_t add_op;
    //
    // op_enum
    EXPECT_EQ( op_enum_t::add, add_op.op_enum() );
    //
    // trace, arg_start, arg_all, ad_type_all, con_vec, par_vec
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
    agraph.op_vec     = std::vector<op_enum_t>( {dom, add} );
    agraph.arg_start  = std::vector<size_t>( {0, 0, 2} );
    agraph.arg_all    = std::vector<size_t>( {0, 0} );
    agraph.ad_type_all = std::vector<ad_type_t>( {par, par} );
    //
    // par_vec[op_index]
    add_op.forward_par(trace, op_index, agraph, con_vec, par_vec);
    //
    // data_ptr
    float* data_ptr = par_vec.at(op_index).data_ptr<float>();
    EXPECT_EQ( data_ptr[0], 2.0 );
    EXPECT_EQ( data_ptr[1], 2.0 );
}
