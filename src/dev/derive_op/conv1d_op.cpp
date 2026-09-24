// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin conv1d_theory usr}
{xrst_spell
    nx
    nw
    ny
    dx
    dw
    pw
    py
    px
}

One Dimensional Cross Correlation Theory
########################################

Function Value
**************
Given the input vector :math:`x \in \mathbb{R}^{nx}`,
the weight vector :math:`w \in \mathbb{R}^{nw}` ,
and the bias tensor :math:`b \in \mathbb{R}` ,
compute the cross correlation vector :math:`y \in \mathbb{R}^{ny}`
where :math:`ny = nx - nw + 1` and

.. math::

    y_k ( x, w, b ) & = b + \sum_{j=0}^{nw-1} x_{k + j} \cdot w_j \\
                    & = b + \sum_{i=k}^{k+nw-1} x_i \cdot w_{i-k} \\

Note that we use the following notation,
:math:`i \in [0, \ldots , nx-1]`,
:math:`j \in [0, \ldots , nw-1]`,
:math:`k \in [0, \ldots , ny-1]`,

Partial Derivatives
*******************

.. math::

    \partial y_k / \partial w_j & = x_{k + j}
    \\
    \partial y_k / \partial x_i & = \begin{cases}
        w_{i-k} & \text{if} \; i \in [k, \cdots, k+nw-1]  \\
        0       & \text{otherwise}
    \end{cases}
    \\
    \partial y_k / \partial b & = 1

Forward Derivatives
*******************
Given the forward derivatives
:math:`dx \in \mathbb{R}^{nx}, dw \in \mathbb{R}^{nw} , db \in \mathbb{R}` ,
the forward derivative of :math:`dy \in \mathbb{R}^{ny}` is given by

.. math::

    dy_k = db + \sum_{j=0}^{nw-1} x_{k+j} \cdot dw_j +
        \sum_{i=k}^{k+nw-1} dx_i \cdot w_{i-k}

Note that this is the sum of x correlated with dw
plus dx correlated with w.

Reverse Partials
****************
We are given the partial derivative of the objective w.r.t the output
:math:`py \in \mathbb{R}^{ny}` .
Reverse mode removes the output from the expression of the objective.
The contributions to the partial w.r.t the weight and bias,
:math:`pw \in \mathbb{R}^{nw} , pb \in \mathbb{R}` ,
are given by

.. math::

    pw_j & = \sum_{k=0}^{ny-1} py_k \cdot x_{k+j} \\
    pb   & = \sum_{k=0}^{ny-1} py_k

Note that pw is x correlated with py and pb is the sum of py.

The contributions to the partial w.r.t the input,
:math:`px \in \mathbb{R}^{nx}` , is given by

.. math::

    px_i & = \sum_{k=0}^{ny-1} py_k \cdot
        \begin{cases}
            w_{i-k} & \text{if} \; i \in [k, \cdots, k+nw-1]  \\
            0       & \text{otherwise}
        \end{cases}
    \\
    px_i & = \sum_{k=i-nw+1}^i py_k \cdot w_{i-k}

In the last equation above :math:`py_k` is zero for
:math:`k < 0` and :math:`k > ny - 1` .
Note that :math:`k` should be in :math:`[0 , \ldots , ny-1 ]` .
We fix this abuse of notation by defining the vector
:math:`\hat{py} \in \mathbb{R}^{nx + nw - 1}`

.. math::

    \hat{py}_l & = \begin{cases}
        0               & \text{if} \; l < nw - 1 \\
        0               & \text{if} \; l > nw - 1 + ny - 1 \\
        py_{l - nw + 1} & \text{otherwise}
    \end{cases}

where :math:`l \in [0 , nx + nw -1 ]` .
Substituting :math:`l = k + nw - 1` in the equation for :math:`px_i` above
we obtain

.. math::

    px_i = \sum_{l=i}^{i+nw-1} \hat{py}_l \cdot w_{i+nw-1-l}

Substituting :math:`j = l - i` we obtain

.. math::

    px_i
    & = \sum_{j=0}^{nw-1} \hat{py}_{i+j} \cdot w_{nw-1-j} \\
    & = \sum_{j=0}^{nw-1} \hat{py}_{i+j} \cdot \hat{w}_j

where :math:`\hat{w}_j = w_{nw-1-j}` .
Note that px is :math:`\hat{py}` correlated with :math:`\hat{w}` .

{xrst_end conv1d_theory}
*/
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/tensor_at_index.hpp>
#include <ad_tensor/no_elements.hpp>
#include <ad_tensor/dev/plus_minus_equal.hpp>
//
namespace ad_tensor { namespace dev { // Begin ad_tensor::dev
// ------------------------------------------------------------------------
// forward_par
template<class TensorType>
void conv1d_op_t<TensorType>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<TensorType>&          par_all
) const {
    //
    // arg_start
    size_t arg_start = agraph.m_arg_start[op_index];
    //
#ifndef NDEBUG
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 6  );
    for(size_t i = 3; i < 6; ++i) {
        assert( agraph.m_arg_type[arg_start+i] == adtype_t::none );
    }
# endif
    //
    // input, weight, bias
    TensorType input  = tensor_at_arg_index(
        arg_start, agraph, con_vec, par_all
    );
    TensorType weight  = tensor_at_arg_index(
        arg_start + 1, agraph, con_vec, par_all
    );
    TensorType bias  = tensor_at_arg_index(
        arg_start + 2, agraph, con_vec, par_all
    );
    //
    // options
    int64_t stride   = int64_t( agraph.m_arg_value[arg_start + 3] );
    int64_t dilation = int64_t( agraph.m_arg_value[arg_start + 4] );
    int64_t groups   = int64_t( agraph.m_arg_value[arg_start + 5] );
    auto options = torch::nn::functional::Conv1dFuncOptions()
        .stride(stride)
        .dilation(dilation)
        .groups(groups);
    //
    // par_all
    par_all[op_index] = conv1d(input, weight, bias, options);
}
template void conv1d_op_t<at::Tensor>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<at::Tensor>&          par_all
) const;
template void conv1d_op_t<adten_t>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<adten_t>&             par_all
) const;
// ------------------------------------------------------------------------
// forward_var
template<class TensorType>
void conv1d_op_t<TensorType>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    vector<TensorType>&          var_all
) const {
    //
    // arg_start
    size_t arg_start = agraph.m_arg_start[op_index];
    //
#ifndef NDEBUG
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 6  );
    for(size_t i = 3; i < 6; ++i) {
        assert( agraph.m_arg_type[arg_start+i] == adtype_t::none );
    }
# endif
    //
    // input, weight, bias
    TensorType input  = tensor_at_arg_index(
        arg_start, agraph, con_vec, par_all, var_all
    );
    TensorType weight  = tensor_at_arg_index(
        arg_start + 1, agraph, con_vec, par_all, var_all
    );
    TensorType bias  = tensor_at_arg_index(
        arg_start + 2, agraph, con_vec, par_all, var_all
    );
    //
    // options
    int64_t stride   = int64_t( agraph.m_arg_value[arg_start + 3] );
    int64_t dilation = int64_t( agraph.m_arg_value[arg_start + 4] );
    int64_t groups   = int64_t( agraph.m_arg_value[arg_start + 5] );
    auto options = torch::nn::functional::Conv1dFuncOptions()
        .stride(stride)
        .dilation(dilation)
        .groups(groups);
    //
    // par_all
    var_all[op_index] = conv1d(input, weight, bias, options);
}
template void conv1d_op_t<at::Tensor>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    vector<at::Tensor>&          var_all
) const;
template void conv1d_op_t<adten_t>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    vector<adten_t>&             var_all
) const;
// ------------------------------------------------------------------------
// forward_der
template<class TensorType>
void conv1d_op_t<TensorType>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          for_der
) const {
    //
    // arg_start
    size_t arg_start = agraph.m_arg_start[op_index];
    //
#ifndef NDEBUG
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 6  );
    for(size_t i = 3; i < 6; ++i) {
        assert( agraph.m_arg_type[arg_start+i] == adtype_t::none );
    }
# endif
    // variable
    adtype_t variable = adtype_t::variable;
    //
    // input_index, input_type, input_zero_der
    size_t   input_index    = agraph.m_arg_value[arg_start];
    adtype_t input_type     = agraph.m_arg_type[arg_start];
    bool     input_zero_der =
        input_type != variable || no_elements( for_der[input_index] );
    //
    // weight_index, weight_type, weight_zero_der
    size_t   weight_index    = agraph.m_arg_value[arg_start + 1];
    adtype_t weight_type     = agraph.m_arg_type[arg_start + 1];
    bool     weight_zero_der =
        weight_type != variable || no_elements( for_der[weight_index] );
    //
    // bias_index, bias_type, bias_zero_der
    size_t   bias_index    = agraph.m_arg_value[arg_start + 2];
    adtype_t bias_type     = agraph.m_arg_type[arg_start + 2];
    bool     bias_zero_der =
        bias_type != variable || no_elements( for_der[bias_index] );
    //
    if( input_zero_der && weight_zero_der && bias_zero_der) {
        return;
    }
    //
    // output_shape
    c10::IntArrayRef output_shape = var_all[op_index].sizes();
    //
    // for_der[op_index]
    TensorType  doutput = TensorType( torch::zeros( output_shape ) );
    if( ! bias_zero_der ) {
        int64_t n_channel_out = output_shape[1];
        doutput += for_der[bias_index].view( {1, n_channel_out, 1}  );
    }
    //
    // for_der[op_index]
    if( input_zero_der && weight_zero_der ) {
        for_der[op_index] = doutput;
        return;
    }
    //
    // options
    int64_t stride   = int64_t( agraph.m_arg_value[arg_start + 3] );
    int64_t dilation = int64_t( agraph.m_arg_value[arg_start + 4] );
    int64_t groups   = int64_t( agraph.m_arg_value[arg_start + 5] );
    auto options = torch::nn::functional::Conv1dFuncOptions()
        .stride(stride)
        .dilation(dilation)
        .groups(groups);
    //
    // no_bias
    TensorType no_bias;
    //
    // doutput
    if( ! input_zero_der ) {
        TensorType weight  = tensor_at_arg_index(
            arg_start + 1, agraph, con_vec, par_all, var_all
        );
        doutput += conv1d( for_der[input_index], weight, no_bias , options );
    }
    if( ! weight_zero_der ) {
        TensorType input  = tensor_at_arg_index(
            arg_start, agraph, con_vec, par_all, var_all
        );
        doutput += conv1d( input, for_der[weight_index], no_bias , options );
    }
    //
    // for_der[op_index]
    for_der[op_index] = doutput;
    return;
}
template void conv1d_op_t<at::Tensor>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          for_der
) const;
template void conv1d_op_t<adten_t>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             for_der
) const;
// ------------------------------------------------------------------------
template<class TensorType>
void conv1d_op_t<TensorType>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          rev_der
) const {
    //
    // arg_start
    size_t arg_start = agraph.m_arg_start[op_index];
    //
#ifndef NDEBUG
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 6  );
    for(size_t i = 3; i < 6; ++i) {
        assert( agraph.m_arg_type[arg_start+i] == adtype_t::none );
    }
# endif
    // variable
    adtype_t variable = adtype_t::variable;
    //
    // input_index, input_type
    size_t   input_index    = agraph.m_arg_value[arg_start];
    adtype_t input_type     = agraph.m_arg_type[arg_start];
    //
    // weight_index, weight_type
    size_t   weight_index    = agraph.m_arg_value[arg_start + 1];
    adtype_t weight_type     = agraph.m_arg_type[arg_start + 1];
    //
    // bias_index, bias_type
    size_t   bias_index    = agraph.m_arg_value[arg_start + 2];
    adtype_t bias_type     = agraph.m_arg_type[arg_start + 2];
    //
    // rev_der[bias_index]
    if( bias_type == variable ) {
        TensorType bias_bar = rev_der[op_index].sum(2);
        plus_equal(rev_der[bias_index], bias_bar);
    }
    if( input_type != variable && weight_type != variable ) {
        return;
    }
    //
    // no_bias
    TensorType no_bias;
    //
    // options
    int64_t stride   = int64_t( agraph.m_arg_value[arg_start + 3] );
    int64_t dilation = int64_t( agraph.m_arg_value[arg_start + 4] );
    int64_t groups   = int64_t( agraph.m_arg_value[arg_start + 5] );
    auto options = torch::nn::functional::Conv1dFuncOptions()
        .stride(stride)
        .dilation(dilation)
        .groups(groups);
    //
    // rev_der[weight_index]
    if( weight_type == variable ) {
        TensorType input  = tensor_at_arg_index(
            arg_start, agraph, con_vec, par_all, var_all
        );
        TensorType weight_bar = conv1d(
            input, rev_der[op_index], no_bias, options
        );
        plus_equal(rev_der[weight_index], weight_bar);
    }
    //
    // rev_der[input_index]
    if( input_type == variable ) {
        //
        // weight, poutput_hat
        TensorType weight  = tensor_at_arg_index(
            arg_start + 1, agraph, con_vec, par_all, var_all
        );
        int64_t kernel_size = weight.sizes()[2];
        auto    pad_options = torch::nn::functional::PadFuncOptions(
            {kernel_size - 1, kernel_size - 1, 0, 0, 0, 0}
        );
        TensorType poutput_hat = torch::nn::functional::pad(
            rev_der[op_index], pad_options
        );
        //
        // input_bar
        TensorType weight_hat = torch::flip(weight, {2});
        TensorType input_bar = conv1d(
            poutput_hat, weight_hat, no_bias, options
        );
        plus_equal(rev_der[input_index], input_bar);
    }
}
template <> void conv1d_op_t<adten_t>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             rev_der
) const {
    // TODO: Change this function to use the TensorType implementation above
    // once the following operators have complete adten_t implementations:
    // pad, flip
    user_assert(false,
    "reverse_der not yet implemented for conv1d with adten_t arguments" );
}
template void conv1d_op_t<at::Tensor>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          rev_der
) const;
// ---------------------------------------------------------------------------
// src_gen
template <> std::string conv1d_op_t<at::Tensor>::src_gen(
    size_t                                                op_index        ,
    const agraph_t&                                       agraph          ,
    bool                                                  variable_agraph ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const {
    //
    // string
    using std::string;
    //
    // arg_start
    size_t arg_start = agraph.m_arg_start[op_index];
    //
#ifndef NDEBUG
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 6  );
    for(size_t i = 3; i < 6; ++i) {
        assert( agraph.m_arg_type[arg_start+i] == adtype_t::none );
    }
# endif
    //
    // input_src
    size_t   input_index   = agraph.m_arg_value[arg_start];
    adtype_t input_adtype  = agraph.m_arg_type[arg_start];
    string   input_src     = tensor_src(input_index, input_adtype);
    //
    // weight_src
    size_t   weight_index   = agraph.m_arg_value[arg_start + 1];
    adtype_t weight_adtype  = agraph.m_arg_type[arg_start + 1];
    string   weight_src     = tensor_src(weight_index, weight_adtype);
    //
    // bias_src
    size_t   bias_index   = agraph.m_arg_value[arg_start + 2];
    adtype_t bias_adtype  = agraph.m_arg_type[arg_start + 2];
    string   bias_src     = tensor_src(bias_index, bias_adtype);
    //
    // target_src
    size_t   target_index = op_index;
    adtype_t target_adtype  =
        variable_agraph ? adtype_t::variable : adtype_t::parameter;
    string   target_src   = tensor_src(target_index, target_adtype);
    //
    // src
    size_t stride   = agraph.m_arg_value[arg_start + 3];
    size_t dilation = agraph.m_arg_value[arg_start + 4];
    size_t groups   = agraph.m_arg_value[arg_start + 5];
    constexpr const char* fmt1 =
R"|({{   auto options = torch::nn::functional::Conv1dFuncOptions()
        .stride({})
        .dilation({})
        .groups({});
)|";
    string src = std::format(fmt1, stride, dilation, groups);
    //
    // src
    constexpr const char* fmt2 =
        "    {} = ad_tensor::conv1d({}, {}, {}, options);";
    src += std::format(fmt2, target_src, input_src, weight_src, bias_src);
    src += "\n}";
    //
    return src;
}
template <> std::string conv1d_op_t<adten_t>::src_gen(
    size_t                                                op_index        ,
    const agraph_t&                                       agraph          ,
    bool                                                  variable_agraph ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const {
    assert(false && "adten_t version of src_gen called for conv1d operator");
}
} } // End ad_tensor::dev
