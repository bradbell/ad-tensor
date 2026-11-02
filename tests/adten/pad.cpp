// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <torch/torch.h>
//
#include <ad_tensor/dev/pad_enum.hpp>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/defined.hpp>
//
namespace {
    //
    // adten_t, adfn_t, vector, pad_enum_t, functional
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::vector;
    using ad_tensor::dev::pad_enum_t;
    namespace functional = torch::nn::functional;
    //
    // p
    vector<at::Tensor> p = { torch::tensor( { {
        {9, 8, 7},
        {6, 5, 4},
        {3, 2, 1},
    } } ) };
    //
    // v
    vector<at::Tensor> v = { torch::tensor( { {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
    } } ) };
    //
    // dv
    vector<at::Tensor> dv = { torch::tensor( { {
        {11, 12, 13},
        {14, 15, 16},
        {17, 18, 19},
    } } ) };
    vector<at::Tensor> py = {
        ad_tensor::not_defined(),
        torch::tensor( { {
            { 1,  2,  3},
            { 4,  5,  6},
            {11, 12, 13},
            {14, 15, 16},
            {17, 18, 19},
            { 7,  8,  9}
        } } ),
    };
    //
    // record_adfn
    adfn_t record_adfn( pad_enum_t pad_enum ) {
        //
        // options_p, options_v, adfn_name
        std::string adfn_name;
        auto options_p = functional::PadFuncOptions( {1, 2, 0, 0} );
        auto options_v = functional::PadFuncOptions( {0, 0, 2, 1} );
        switch( pad_enum ) {
            //
            case pad_enum_t::constant:
            options_p.mode(torch::kConstant);
            options_v.mode(torch::kConstant);
            options_p.value(10);
            options_v.value(10);
            adfn_name = "constant_padding";
            break;
            //
            case pad_enum_t::replicate:
            options_p.mode(torch::kReplicate);
            options_v.mode(torch::kReplicate);
            adfn_name = "replicate_padding";
            break;
            //
            case pad_enum_t::reflect:
            options_p.mode(torch::kReflect);
            options_v.mode(torch::kReflect);
            adfn_name = "reflect_padding";
            break;
            //
            default:
            assert( false );
            break;
        }
        //
        // av, ap
        auto [av, ap] = adten_t::start_recording(v, p);
        //
        // ay
        vector<adten_t> ay;
        ay.push_back( ad_tensor::pad(ap[0], options_p) );
        ay.push_back( ad_tensor::pad(av[0], options_v) );
        //
        // f
        adfn_t adfn = adten_t::stop_recording(ay, adfn_name);
        //
        return adfn;
    }
}
//
TEST(tests_adten, pad_constant)  {
    //
    adfn_t f = record_adfn( pad_enum_t::constant );
    //
    // y
    vector<at::Tensor> par_all = f.forward_par(p);
    vector<at::Tensor> var_all = f.forward_var(v, par_all);
    vector<at::Tensor> y = f.get_range(var_all, par_all);
    // ----------------------------------------------------------------------
    // check function values
    // ----------------------------------------------------------------------
    at::Tensor check_p = torch::tensor( { {
        {10, 9, 8, 7, 10, 10},
        {10, 6, 5, 4, 10, 10},
        {10, 3, 2, 1, 10, 10},
    } } );
    EXPECT_TRUE( y[0].equal( check_p ) );
    //
    // check y[1]
    at::Tensor check_v = torch::tensor( { {
        {10, 10, 10},
        {10, 10, 10},
        { 1,  2,  3},
        { 4,  5,  6},
        { 7,  8,  9},
        {10, 10, 10},
    } } );
    EXPECT_TRUE( y[1].equal( check_v ) );
    // ------------------------------------------------------------------------
    // check forward mode
    // ------------------------------------------------------------------------
    vector<at::Tensor> dy = f.forward_der(dv, var_all, par_all);
    EXPECT_TRUE( ad_tensor::not_defined( dy[0] ) );
    at::Tensor check_dy = torch::tensor( { {
        { 0,  0,  0},
        { 0,  0,  0},
        {11, 12, 13},
        {14, 15, 16},
        {17, 18, 19},
        { 0,  0,  0},
    } } );
    EXPECT_TRUE( dy[1].equal( check_dy ) );
    // ------------------------------------------------------------------------
    // check reverse mode
    // ------------------------------------------------------------------------
    vector<at::Tensor> px = f.reverse_der(py, var_all, par_all);
    at::Tensor check_px = torch::tensor( { {
        {11, 12, 13},
        {14, 15, 16},
        {17, 18, 19},
    } } );
    EXPECT_TRUE( px[0].equal( check_px ) );
}
TEST(tests_adten, pad_replicate)  {
    //
    // f
    adfn_t f = record_adfn( pad_enum_t::replicate );
    //
    // y
    vector<at::Tensor> par_all = f.forward_par(p);
    vector<at::Tensor> var_all = f.forward_var(v, par_all);
    vector<at::Tensor> y = f.get_range(var_all, par_all);
    // ----------------------------------------------------------------------
    // check function values
    // ----------------------------------------------------------------------
    at::Tensor check_p = torch::tensor( { {
        {9, 9, 8, 7, 7, 7},
        {6, 6, 5, 4, 4, 4},
        {3, 3, 2, 1, 1, 1},
    } } );
    EXPECT_TRUE( y[0].equal( check_p ) );
    //
    // check y[1]
    at::Tensor check_v = torch::tensor( { {
        {1, 2, 3},
        {1, 2, 3},
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {7, 8, 9},
    } } );
    EXPECT_TRUE( y[1].equal( check_v ) );
    // ------------------------------------------------------------------------
    // check forward mode
    // ------------------------------------------------------------------------
    vector<at::Tensor> dy = f.forward_der(dv, var_all, par_all);
    EXPECT_TRUE( ad_tensor::not_defined( dy[0] ) );
    at::Tensor check_dy = torch::tensor( { {
        {11, 12, 13},
        {11, 12, 13},
        {11, 12, 13},
        {14, 15, 16},
        {17, 18, 19},
        {17, 18, 19},
    } } );
    EXPECT_TRUE( dy[1].equal( check_dy ) );
    // ------------------------------------------------------------------------
    // check reverse mode
    // ------------------------------------------------------------------------
    vector<at::Tensor> px = f.reverse_der(py, var_all, par_all);
    at::Tensor check_px = torch::tensor( { {
        {11+4+1, 12+5+2, 13+6+3},
        {    14,     15,     16},
        {  17+7,   18+8,   19+9},
    } } );
    EXPECT_TRUE( px[0].equal( check_px ) );
}
TEST(tests_adten, pad_reflect)  {
    //
    adfn_t f = record_adfn( pad_enum_t::reflect );
    //
    // y
    vector<at::Tensor> par_all = f.forward_par(p);
    vector<at::Tensor> var_all = f.forward_var(v, par_all);
    vector<at::Tensor> y = f.get_range(var_all, par_all);
    // ----------------------------------------------------------------------
    // check function values
    // ----------------------------------------------------------------------
    at::Tensor check_p = torch::tensor( { {
        { 8, 9, 8, 7,  8,  9},
        { 5, 6, 5, 4,  5,  6},
        { 2, 3, 2, 1,  2,  3},
    } } );
    EXPECT_TRUE( y[0].equal( check_p ) );
    //
    // check y[1]
    at::Tensor check_v = torch::tensor( { {
        { 7,  8,  9},
        { 4,  5,  6},
        { 1,  2,  3},
        { 4,  5,  6},
        { 7,  8,  9},
        { 4,  5,  6},
    } } );
    EXPECT_TRUE( y[1].equal( check_v ) );
    // ------------------------------------------------------------------------
    // check forward mode
    // ------------------------------------------------------------------------
    vector<at::Tensor> dy = f.forward_der(dv, var_all, par_all);
    EXPECT_TRUE( ad_tensor::not_defined( dy[0] ) );
    at::Tensor check_dy = torch::tensor( { {
        {17, 18, 19},
        {14, 15, 16},
        {11, 12, 13},
        {14, 15, 16},
        {17, 18, 19},
        {14, 15, 16},
    } } );
    EXPECT_TRUE( dy[1].equal( check_dy ) );
    // ------------------------------------------------------------------------
    // check reverse mode
    // ------------------------------------------------------------------------
    vector<at::Tensor> px = f.reverse_der(py, var_all, par_all);
    at::Tensor check_px = torch::tensor( { {
            {    11,     12,     13},
            {14+4+7, 15+5+8, 16+6+9},
            {  17+1,   18+2,   19+3},
    } } );
    EXPECT_TRUE( px[0].equal( check_px ) );
}
// END_CPP
