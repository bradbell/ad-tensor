#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin_parent atom usr}

Atomic Functions
################

{xrst_toc_table
    src/adten/call_atom.cpp
}

{xrst_end atom}
-------------------------------------------------------------------------------
{xrst_begin atom_dev dev}

Atomic Functions Developer Documentation
########################################

{xrst_toc_table
    src/adten/call_atom.cpp
}

{xrst_end atom_dev}
-------------------------------------------------------------------------------
{xrst_begin atom_callback usr}
{xrst_spell
    rng
    adten
    numel
}

The Atomic Callback Class atom_callback_t
#########################################

atom_callback
*************
We use atom_callback below for an atom_callback_t
object created using its default constructor.

atom_f
******
We use atom_f below for the function corresponding to atom_callback; i.e.

    range = atom_f(domain)

where domain and range are vectors of tensors.

call_info
*********
This value is passed through to the callback functions
from :ref:`call_atom@call_info` in the call to this atomic function.
Each atomic function callback
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
is the vector of the domain tensors for this atomic function call.

set
***

name
====
After the atom_callback_t constructor,
the name for the atomic function is empty.
It must set this with a call of the form

trace
=====
After the atom_callback_t constructor, the trace flag is false.
It can be changed with a call of the form

Functions
=========

.. csv-table::
    :header-rows: 1

    callback,       Required when this atomic is used with
    long_name       never
    depend,         always
    forward,        always
    forward_der,    :ref:`adfn_forward_der-name` with TensorType at::Tensor
    reverse_der,    :ref:`adfn_reverse_der-name` with TensorType at::Tensor
    ad_forward_der, :ref:`adfn_forward_der-name` with TensorType adten
    ad_reverse_der, :ref:`adfn_reverse_der-name` with TensorType adten

The default long_name callback function returns the name chosen by set_name.
After the atom_callback_t constructor,
all the other callback functions are null.
of these callback function that is used must be set by a call of the form
{xrst_code cpp}
    atom_callback.set_function(const function_t& function)
{xrst_code}
where function is the callback function; e.g.,
{xrst_code cpp}
    atom_callback.set_depend(const depend_t& depend)
{xrst_code}

get
***
All of the set functions have a corresponding get function:
The values chosen by set_name and set_trace are in the get_options return.

long_name
*********
This returns a long name for this atomic function that can include
information determined by the value of call_info.
The default version of this callback just returns the name in its
options argument which is the same as :ref:`atom_callback@set@name` above.

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

{xrst_end atom_callback}
-------------------------------------------------------------------------------
{xrst_begin atom_global usr}

Global Object That holds All Atomic Callbacks
#############################################
This global object holds all the atomic functions.

atom_global
***********
{xrst_literal ,
    BEGIN_SINGLETON, END_SINGLETON
}

store
*****
{xrst_literal ,
    BEGIN_STORE, END_STORE
}
A call to store will wait until it can lock out any other calls to
store or get.

atom_callback
=============
is the callback information for this atomic function.

atom_id
=======
Is the identifier for this atomic function.

get_callback
************
A call to get will wait until it can lock out any calls to store.


{xrst_end atom_global}
*/
#include <torch/torch.h>
#include <ad_tensor/vector.hpp>
#include <ad_tensor/sparsity.hpp>
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/options.hpp>
#include <ad_tensor/base_atom.hpp>
//
namespace ad_tensor {  // BEGIN_AD_TENSOR_NAMESPACE
//
// atom_global_t
class atom_global_t {
private:
    std::shared_mutex                          m_rw_mutex;
    vector< std::unique_ptr<base_atom_t> >     m_base_vec;
    //
    // default constructor
    atom_global_t(void)
    { }
public:
    //
    // automatic copy constructor
    atom_global_t(const atom_global_t&) = delete;
    //
    //  automatic assignment operator
    void operator=(const atom_global_t&) = delete;
    //
    // BEGIN_SINGLETON
    // atom_global = atom_global_t::singleton()
    static atom_global_t& singleton(void);
    // END_SINGLETON
    //
    // BEGIN_STORE
    // atom_id = atom_global.store(base_atom_ptr)
    size_t store( std::unique_ptr<base_atom_t>& base_atom_ptr );
    // END_STORE
    //
    // BEGIN_GET_BASE_ATOM
    const base_atom_t& get_base_atom(size_t atom_id);
    // END_GET_BASE_ATOM
};

} // END_AD_TENSOR_NAMESPACE
