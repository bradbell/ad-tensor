// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin conv_transpose1d_theory usr}
{xrst_spell
    rcl
    ny
    nx
    nw
    dx
    dw
    px
    pw
    py
}

One Dimensional Transposed Cross Correlation Theory
###################################################

Function Value
**************
Given the input vector :math:`x \in \mathbb{R}^{nx}`,
the weight vector :math:`w \in \mathbb{R}^{nw}` ,
and the bias tensor :math:`b \in \mathbb{R}` .
The transposed correlation of x with w,b
is the vector :math:`y \in \mathbb{R}^{ny}`
where :math:`ny = nx + nw - 1` and

.. math::

    y_i ( x, w, b ) = b +
        \sum_{ k = \max(0 , i - nw + 1) }^{\min(i, nx - 1)} x_k \cdot w_{i-k}

Note that we use the following notation:
:math:`i \in [0, \ldots , ny-1]` ,
:math:`k \in [0, \ldots , nx-1]` .

Changing indices to :math:`j = i - k` .
Note that :math:`j \in [0 , \ldots , nw - 1]` and

.. math::

    \begin{array}{rcl}
        k \ge 0          & \Leftrightarrow & j \le i      \\
        k \ge i - nw + 1 & \Leftrightarrow & j \le nw - 1 \\
        k \le i          & \Leftrightarrow & j \ge 0      \\
        k \le nx - 1     & \Leftrightarrow & j \ge i - nx + 1
    \end{array}


.. math::

    y_i ( x, w, b ) = b +
        \sum_{j = \max(0, i - nx + 1) }^{\min(i, nw - 1)} x_{i-j} \cdot w_j


Partial Derivatives
*******************

.. math::

    \partial y_i  / \partial x_k
    &= \begin{cases}
        w_{i-k} & \text{if} \; i - nw + 1 \le k \; \text{and} \; k \le i \\
        0       & \text{otherwise}
    \end{cases}
    \\
    \partial y_i  / \partial w_j
    &= \begin{cases}
        x_{i-j} & \text{if} \; i - nx + 1 \le j \; \text{and} \; j \le i \\
        0       & \text{otherwise}
    \end{cases}
    \\
    \partial y_i  / \partial b &= 1


Forward Derivatives
*******************
Given the forward derivatives
:math:`dx \in \mathbb{R}^{nx}, dw \in \mathbb{R}^{nw} , db \in \mathbb{R}` ,
the forward derivative of :math:`dy \in \mathbb{R}^{ny}` is given by

.. math::

    dy_i = db
    + \sum_{k = \max(0 , i - nw + 1)}^{\min(i, nx - 1)} dx_k \cdot w_{i-k}
    + \sum_{j = \max(0, i - nx + 1)}^{ \min(i, nw - 1)} x_{i-j} \cdot dw_j

Note that this is the sum of dx transpose correlated with w,0
plus x transpose correlated with dw,0.


Reverse Partials
****************
We are given the partial derivative of the objective w.r.t the output
:math:`py \in \mathbb{R}^{ny}` .
Reverse mode removes the output from the expression of the objective by adding
the contributions to the partial w.r.t the input, weight and bias,
:math:`px \in \mathbb{R}^{nx}` ,
:math:`pw \in \mathbb{R}^{nw} ,
pb \in \mathbb{R}` .
The contribution w.r.t the input is given by

.. math::

    px_k
    &= \sum_{i = 0}^{ny-1} py_i \cdot \partial y_i / \partial x_k  \\
    &= \sum_{i = k}^{k + nw - 1} py_i \cdot w_{i-k}

Note that this is py correlated with w,0.
The contribution w.r.t. the weight is given by

.. math::

    pw_j
    &= \sum_{i = 0}^{ny-1} py_i \cdot \partial y_i / \partial w_j  \\
    &= \sum_{i = j}^{j + nx - 1} py_i \cdot x_{i-j}

Note that this is py correlated with x,0.
The contribution w.r.t. the bias is given by

.. math::

    pb
    &= \sum_{i=0}^{ny-1} py_i \cdot \partial y_i / \partial b \\
    &= \sum_{i=0}^{ny-1} py_i


{xrst_end conv_transpose1d_theory}
*/
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/tensor_at_index.hpp>
#include <ad_tensor/defined.hpp>
#include <ad_tensor/dev/plus_minus_equal.hpp>
//
// set options
#define SET_OPTIONS \
    auto options = torch::nn::functional::ConvTranspose1dFuncOptions(); \
    { \
        int64_t stride   = int64_t( agraph.m_arg_value[arg_start + 3] ); \
        int64_t dilation = int64_t( agraph.m_arg_value[arg_start + 4] ); \
        int64_t groups   = int64_t( agraph.m_arg_value[arg_start + 5] ); \
        options.stride(stride); \
        options.dilation(dilation); \
        options.groups(groups); \
    }
//
// check arguments
#ifdef NDEBUG
    #define CHECK_ARGUMENTS
#else
    #define CHECK_ARGUMENTS \
    {   size_t arg_start = agraph.m_arg_start[op_index]; \
        size_t arg_end   = agraph.m_arg_start[op_index+1]; \
        size_t n_arg = arg_end - arg_start; \
        assert( n_arg == 6  ); \
        for(size_t i = 3; i < 6; ++i) { \
            assert( agraph.m_arg_type[arg_start+i] == adtype_t::none ); \
        } \
    }
#endif
//
namespace ad_tensor { namespace dev { // Begin ad_tensor::dev
// ------------------------------------------------------------------------
// forward_par
template<class TensorType>
void conv_transpose1d_op_t<TensorType>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<TensorType>&          par_all
) const {
    CHECK_ARGUMENTS
    //
    // arg_start
    size_t arg_start = agraph.m_arg_start[op_index];
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
    SET_OPTIONS
    //
    // par_all
    par_all[op_index] = conv_transpose1d(input, weight, bias, options);
}
template void conv_transpose1d_op_t<at::Tensor>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<at::Tensor>&          par_all
) const;
template void conv_transpose1d_op_t<adten_t>::forward_par(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    vector<adten_t>&             par_all
) const;
// ------------------------------------------------------------------------
// forward_var
template<class TensorType>
void conv_transpose1d_op_t<TensorType>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    vector<TensorType>&          var_all
) const {
    CHECK_ARGUMENTS
    //
    // arg_start
    size_t arg_start = agraph.m_arg_start[op_index];
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
    SET_OPTIONS
    //
    // par_all
    var_all[op_index] = conv_transpose1d(input, weight, bias, options);
}
template void conv_transpose1d_op_t<at::Tensor>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    vector<at::Tensor>&          var_all
) const;
template void conv_transpose1d_op_t<adten_t>::forward_var(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    vector<adten_t>&             var_all
) const;
// ------------------------------------------------------------------------
// forward_der
// forward_der
template<class TensorType>
void conv_transpose1d_op_t<TensorType>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          for_der
) const {
    CHECK_ARGUMENTS
    //
    // arg_start
    size_t arg_start = agraph.m_arg_start[op_index];
    //
    // variable
    adtype_t variable = adtype_t::variable;
    //
    // input_index, input_type, input_zero_der
    size_t   input_index    = agraph.m_arg_value[arg_start];
    adtype_t input_type     = agraph.m_arg_type[arg_start];
    bool     input_zero_der =
        input_type != variable || not_defined( for_der[input_index] );
    //
    // weight_index, weight_type, weight_zero_der
    size_t   weight_index    = agraph.m_arg_value[arg_start + 1];
    adtype_t weight_type     = agraph.m_arg_type[arg_start + 1];
    bool     weight_zero_der =
        weight_type != variable || not_defined( for_der[weight_index] );
    //
    // bias_index, bias_type, bias_zero_der
    size_t   bias_index    = agraph.m_arg_value[arg_start + 2];
    adtype_t bias_type     = agraph.m_arg_type[arg_start + 2];
    bool     bias_zero_der =
        bias_type != variable || not_defined( for_der[bias_index] );
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
    SET_OPTIONS
    //
    // no_bias
    TensorType no_bias;
    //
    // doutput
    if( ! input_zero_der ) {
        TensorType weight  = tensor_at_arg_index(
            arg_start + 1, agraph, con_vec, par_all, var_all
        );
        doutput += conv_transpose1d(
            for_der[input_index], weight, no_bias , options
        );
    }
    if( ! weight_zero_der ) {
        TensorType input  = tensor_at_arg_index(
            arg_start, agraph, con_vec, par_all, var_all
        );
        doutput += conv_transpose1d(
            input, for_der[weight_index], no_bias , options
        );
    }
    //
    // for_der[op_index]
    for_der[op_index] = doutput;
    return;
}
template void conv_transpose1d_op_t<at::Tensor>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          for_der
) const;
template void conv_transpose1d_op_t<adten_t>::forward_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             for_der
) const;
// ------------------------------------------------------------------------
template <class TensorType>
void conv_transpose1d_op_t<TensorType>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<TensorType>&    par_all     ,
    const vector<TensorType>&    var_all     ,
    vector<TensorType>&          rev_der
) const {
    CHECK_ARGUMENTS
    //
    // arg_start
    size_t arg_start = agraph.m_arg_start[op_index];
    //
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
    // stride, dilation, group
    int64_t stride   = int64_t( agraph.m_arg_value[arg_start + 3] ); \
    int64_t dilation = int64_t( agraph.m_arg_value[arg_start + 4] ); \
    int64_t groups   = int64_t( agraph.m_arg_value[arg_start + 5] ); \
    //
    // rev_der[weight_index]
    if( weight_type == variable ) {
        //
        auto options = torch::nn::functional::Conv1dFuncOptions();
        options.stride(stride);
        options.dilation(dilation);
        options.groups(groups);
        //
        TensorType input  = tensor_at_arg_index(
            arg_start, agraph, con_vec, par_all, var_all
        );
        TensorType weight_bar = conv1d(
            rev_der[op_index], input, no_bias, options
        );
        plus_equal(rev_der[weight_index], weight_bar);
    }
    //
    // rev_der[input_index]
    if( input_type == variable ) {
        //
        auto options = torch::nn::functional::Conv1dFuncOptions();
        options.stride(stride);
        options.dilation(dilation);
        options.groups(groups);
        //
        TensorType weight  = tensor_at_arg_index(
            arg_start + 1, agraph, con_vec, par_all, var_all
        );
        //
        TensorType input_bar = ad_tensor::conv1d(
            rev_der[op_index], weight, no_bias, options
        );
        //
        plus_equal(rev_der[input_index], input_bar);
    }
}
template void conv_transpose1d_op_t<adten_t>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<adten_t>&       par_all     ,
    const vector<adten_t>&       var_all     ,
    vector<adten_t>&             rev_der
) const;
template void conv_transpose1d_op_t<at::Tensor>::reverse_der(
    size_t                       op_index    ,
    const agraph_t&              agraph      ,
    const vector<at::Tensor>&    con_vec     ,
    const vector<at::Tensor>&    par_all     ,
    const vector<at::Tensor>&    var_all     ,
    vector<at::Tensor>&          rev_der
) const;
// ---------------------------------------------------------------------------
// src_gen
template <> std::string conv_transpose1d_op_t<at::Tensor>::src_gen(
    size_t                                                op_index        ,
    adtype_t                                              op_adtype       ,
    const agraph_t&                                       agraph          ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const {
    CHECK_ARGUMENTS
    //
    // string
    using std::string;
    //
    // arg_start
    size_t arg_start = agraph.m_arg_start[op_index];
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
    string target_src = tensor_src(op_index, op_adtype);
    //
    // src
    size_t stride   = agraph.m_arg_value[arg_start + 3];
    size_t dilation = agraph.m_arg_value[arg_start + 4];
    size_t groups   = agraph.m_arg_value[arg_start + 5];
    constexpr const char* fmt1 =
R"|({{   auto options = torch::nn::functional::ConvTranspose1dFuncOptions()
        .stride({})
        .dilation({})
        .groups({});
)|";
    string src = std::format(fmt1, stride, dilation, groups);
    //
    // src
    constexpr const char* fmt2 =
        "    {} = ad_tensor::conv_transpose1d({}, {}, {}, options);";
    src += std::format(fmt2, target_src, input_src, weight_src, bias_src);
    src += "\n}";
    //
    return src;
}
template <> std::string conv_transpose1d_op_t<adten_t>::src_gen(
    size_t                                                op_index        ,
    adtype_t                                              op_adtype       ,
    const agraph_t&                                       agraph          ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src
) const {
    assert(false &&
        "adten_t version of src_gen called for conv_transpose1d operator"
    );
    return "";
}
// ---------------------------------------------------------------------------
} } // End ad_tensor::dev
