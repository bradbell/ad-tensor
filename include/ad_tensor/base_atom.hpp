#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin base_atom usr}
{xrst_spell
    rng
    adten
    numel
}

The Atomic Function Base Class
##############################

atom_f
******
We use atom_f below for the function corresponding to a class
derived from this base class:

    range = atom_f(domain)

where domain and range are vectors of tensors.

name
****
After the base class constructor,
the name for the atomic function is empty.
This is used for error reporting an a derived class should
set it to a useful value using set_name.

trace
*****
After the base class constructor,
the trace flag for the atomic function is false.
It can be set using set_trace and a derived virtual function
may use this flag as a signal to generate debugging output.

call_info
*********
This value is passed through to the virtual functions
from :ref:`call_atom@call_info` in the call to this atomic function.
Each derived version of the virtual functions
can decided what to do based on the value of call_info.

rng_used
********
If rng_used is the empty vector, all of the range values are used.
Otherwise this vector has the same size as :ref:`call_atom@arange`
in the corresponding call this atomic function.
If rng_used[i] is true,
the return tensor with index i is used and must be calculated.
If it is false,
the return tensor with index i is not used and need not be calculated.

domain
******
is the vector of the domain tensors containing the domain values.

range
*****
is the vector of the range tensors containing the range values.

dom_der
*******
is the vector of the domain tensors containing the
domain directional derivative.

rng_der
*******
is the vector of the range tensors containing the
range directional derivative.

Virtual Functions
*****************

.. csv-table::
    :header-rows: 1

    function,       Required when this atomic is used with
    long_name       never
    depend,         always
    forward,        always
    forward_der,    :ref:`adfn_forward_der-name` with TensorType at::Tensor
    reverse_der,    :ref:`adfn_reverse_der-name` with TensorType at::Tensor
    ad_forward_der, :ref:`adfn_forward_der-name` with TensorType adten
    ad_reverse_der, :ref:`adfn_reverse_der-name` with TensorType adten

Prototype
=========
{xrst_literal,
    BEGIN_CLASS, END_CLASS
}


long_name
*********
The default version of this function returns the name described above.
A derived long_name should be defined when a description of the function
depends on to the value of call_info.

depend
******
This returns a dependency :ref:`sparsity-name` pattern
for this atomic function.
If the tensor with range index i depends on the tensor with domain index j,
then (i,j) is in the sparsity pattern for this atomic function.

forward
*******
This computes the atomic function range values
as a function of its domain values; i.e.,

    range = atom_f(domain)


forward_der
***********
This computes the atomic function range derivative tensors
as a function of its domain tensors and domain derivative tensors; i.e

    rng_der = atom_f'(domain) * dom_der

It is suggested that the AD version of forward_der call an atomic function
so the values for the calculations are not in the tape being recorded.

reverse_der
***********
This computes the atomic function domain derivative tensors
as a function of its domain tensors and range derivative tensors; i.e

    dom_der = rng_der * atom_f'(domain)

#.  If rng_der[i] is empty, rng_der[i].numel() is zero,
    rng_der[i] should be treated as zero with the proper dimensions
    for the i-th range component of the atomic function.

#.  If you know dom_der[j] is zero based on the empty input tensors,
    you can return an empty dom_der[j],
    instead of zero with the proper dimensions for dom_der[j] .
    This avoids the memory for the operations that use this return value.

#.  It is suggested that the AD version of forward_der call a different
    atomic function so the values for the calculations are not in the
    tape being recorded.

{xrst_end base_atom}
*/
#include <torch/torch.h>
#include <ad_tensor/vector.hpp>
#include <ad_tensor/adten.hpp>
#include <string>
#include <optional>
// BEGIN_CLASS
namespace ad_tensor { class base_atom_t {
private:
    std::string m_name;
    bool        m_trace;
public:
    // ctor
    base_atom_t(void);
    //
    // name
    void set_name(const std::string& trace);
    const std::string& get_name(void) const;
    //
    // trace
    void set_trace(bool trace);
    bool get_trace(void) const;
    //
    // long_name
    virtual std::string long_name(size_t call_info) const;
    //
    // depend
    virtual std::optional<sparsity_t> depend(size_t call_info) const;
    //
    // forward
    virtual std::optional< vector<at::Tensor> > forward(
        size_t                      call_info,
        const vector<bool>&         rng_used,
        const vector<at::Tensor>&   domain
    ) const;
    //
    // forward_der
    virtual std::optional< vector<at::Tensor> > forward_der(
        size_t                      call_info,
        const vector<bool>&         rng_used,
        const vector<at::Tensor>&   domain,
        const vector<at::Tensor>&   dom_der
    ) const;
    virtual std::optional< vector<adten_t> > forward_der(
        size_t                      call_info,
        const vector<bool>&         rng_used,
        const vector<adten_t>&      domain,
        const vector<adten_t>&      dom_der
    ) const;
    //
    // reverse_der
    virtual std::optional< vector<at::Tensor> > reverse_der(
        size_t                     call_info,
        const vector<bool>&        rng_used,
        const vector<at::Tensor>&  domain,
        const vector<at::Tensor>&  dom_der
    ) const;
    virtual std::optional< vector<adten_t> > reverse_der(
        size_t                     call_info,
        const vector<bool>&        rng_used,
        const vector<adten_t>&     domain,
        const vector<adten_t>&     dom_der
    ) const;
}; }
// END_CLASS
