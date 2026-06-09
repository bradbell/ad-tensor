// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <ad_tensor/adfn.hpp>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/to_string.hpp>
TEST(tests, adten_solve_op)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::options_t;
    using at::Tensor;
    using ad_tensor::vector;
    //
    // options
    options_t options;
    //
    // x
    double x0 = 0.5, x1 = 1.0, x2 = 1.5, x3 = 2.0;
    vector<Tensor> x;
    x.push_back( torch::tensor( {
        {x0, x1},
        {x2, x3}
    } ) );
    /*
    inv(x) = {
        {x3, -x1},
        {-x2, x0}
    } / det(x0, x1, x2, x3)
    */
    // p
    vector<Tensor> p;
    double p0 = 1.0, p1 = 2.0, p2 = 3.0, p3 = 4.0;
    p.push_back( torch::tensor( {
        {p0, p1},
        {p2, p3}
    } ) );
    //
    // ap, ax
    auto [ap, ax] = adten_t::start_recording(p, x);
    //
    // ay
    vector<adten_t> ay;
    ay.push_back( ad_tensor::linalg_solve(ax[0], ap[0]) );
    //
    // y = f(p)
    adfn_t f = adten_t::stop_recording(ay);
    //
    // par_all, var_all
    vector<Tensor> par_all = f.forward_par(p, options);
    vector<Tensor> var_all = f.forward_var(par_all, x, options);
    //
    // y
    vector<Tensor> y = f.get_range(par_all, var_all);
    //
    // det
    double det = x0 * x3 - x1 * x2;
    //
    // y0_times_det
    Tensor y0_times_det = torch::tensor( {
        {  x3 * p0 - x1 * p2,   x3 * p1 - x1 * p3},
        {- x2 * p0 + x0 * p2, - x2 * p1 + x0 * p3}
    } );
    //
    // inv(x) * p
    bool equal = y[0].equal( y0_times_det / det);
    EXPECT_TRUE( equal );
    //
    Tensor check;
    //
    // dx, dy
    vector<Tensor> dx(1), dy(1);
    dx[0]  = torch::tensor( { {1.0, 0.0}, {0.0, 0.0} } );
    dy     = f.forward_der(par_all, var_all, dx, options);
    check  = torch::tensor( { {0.0, 0.0}, {p2,  p3} } ) / det;
    check -= y0_times_det * x3 / (det * det);
    bool close = dy[0].allclose( check );
    EXPECT_TRUE( close );
}
// END_CPP
