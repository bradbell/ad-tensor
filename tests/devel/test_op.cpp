// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <ad_tensor/devel/op_enum.hpp>
#include <ad_tensor/devel/ad_type.hpp>
#include <ad_tensor/devel/op_base.hpp>
//
// empty namespace
namespace {
    //
    // op_enum_t, ad_type_t
    using ad_tensor::devel::op_enum_t;
    using ad_tensor::devel::ad_type_t;
    using ad_tensor::devel::op_base_t;
    //
    // vector
    using std::vector;
    //
    // test_op_t
    class test_op_t : public  op_base_t {
    public:
        //
        // op_enum
        op_enum_t op_enum(void) const override {
            return op_enum_t::test;
        }
        //
        // n_arg
        size_t n_arg(
            size_t                 offset      ,
            const  vector<size_t>& arg_all     ,
            const  vector<ad_type_t>& ad_type_all ) const override
        {   return 2; }
    };
}

TEST(tests_devel, op_base)  {
    //
    // test_op
    test_op_t test_op;
    //
    // op_enum
    op_enum_t op_enum = test_op.op_enum();
    EXPECT_EQ( op_enum,  op_enum_t::test );
    //
    // n_arg
    size_t offset = 0;
    vector<size_t> arg_all;
    vector<ad_type_t> ad_type_all;
    size_t n_arg = test_op.n_arg(offset, arg_all, ad_type_all);
    EXPECT_EQ( n_arg,    2);
}
