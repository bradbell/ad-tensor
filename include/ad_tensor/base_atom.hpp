#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin_parent base_atom usr}

The Atomic Function Base Class
##############################
An atomic function is defined by a derived class
that overrides of the virtual functions in base_atom_t.

{xrst_end base_atom}
-------------------------------------------------------------------------------
{xrst_begin atom_non_virtual usr}

The Non-Virtual Functions in base_atom_t
########################################

atom_base
*********
{xrst_literal ,
    BEGIN_CTOR, END_CTOR
}
This initializes the name as empty and the trace flag as false.

name
****
{xrst_literal ,
    BEGIN_NAME, END_NAME
}

trace
*****
{xrst_literal ,
    BEGIN_TRACE, END_TRACE
}

{xrst_end atom_non_virtual}
------------------------------------------------------------------------------
{xrst_begin atom_destructor usr}

Destructor
##########

Discussion
**********
The base class destructor is virtual so that
a base class pointer can call the derived class destructor.

Code
****
{xrst_literal ,
    BEGIN_DESTRUCTOR, END_DESTRUCTOR
}

{xrst_end atom_destructor}
------------------------------------------------------------------------------
{xrst_begin atom_depend usr}

The Dependency Sparsity Pattern
###############################

Syntax
******
{xrst_code hpp}
    depend = base_atom.depend().value()
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_DEPEND, END_DEPEND
}
This virtual function is first used during make a :ref:`call_atom-name`
for a derived atomic function.
If you do not over ride this virtual function its return
will have has_value() equal to false.
You should also use this return when an error occurs during
the evaluation of the sparsity pattern for the derived atomic function.

depend
******
If the i-th range tensor depends on the j-th domain tensor
then the pair (i,j) must be in the depend sparsity pattern.

{xrst_end atom_depend}
------------------------------------------------------------------------------
{xrst_begin atom_forward usr}
{xrst_spell
    rng
}

Forward Function Evaluation
###########################

Syntax
******
{xrst_code hpp}
    range = base_atom.forward(rng_used, domain).value()
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_FORWARD_FUN, END_FORWARD_FUN
}
This virtual function is first used during make a :ref:`call_atom-name`
for a derived atomic function.
If you do not over ride this virtual function its return
will have has_value() equal to false.
You should also use this return when an error occurs during
the evaluation of the sparsity pattern for the derived atomic function.

rng_used
********
If rng_used is the empty vector, all of the range values are used.
Otherwise this vector has the same size as :ref:`call_atom@arange`
in the corresponding call this atomic function.
If rng_used[i] is true (false),
the return tensor with index i is used and must be calculated
(is not used).
Either rng_used is empty, or at least one of its elements is true.

domain
******
is the vector of the domain tensors containing the domain values.
Given rng_used and :ref:`atom_depend-name` it is possible to determine
that some domain values are not needed.
These domain values may be changed to :ref:`empty_at_ten-name` .

range
*****
is the vector of the range tensors containing the range values.

atom_f
******
We use the notation atom_f for the mathematical function defined by

    range = atom_f(domain)

{xrst_end atom_forward}
------------------------------------------------------------------------------
{xrst_begin atom_forward_der usr}
{xrst_spell
    rng
    adten
}

Forward Derivative Evaluation
#############################

Syntax
******
{xrst_code hpp}
rng_der = base_atom.forward_der(rng_used, domain, dom_der).value()
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_FORWARD_DER, END_FORWARD_DER
}
This virtual function is used if:

#.  You make a :ref:`call_atom-name` to this atomic function.
#.  The call becomes part of an AD function.
#.  The :ref:`forward derivative<adfn_forward_der-name>`
    of the AD function is evaluated.

If you do not over ride this virtual function its return
will have has_value() equal to false.
You should also use this case when an error occurs during
the evaluation of this derivative of the atomic function.

If dom_der[j].numel() is zero, then dom_der[j] represents a zero tensor
with the same shape as domain[j] and calculations that use this value
can be skipped.

If domain in the AD function forward_der evaluation is a vector of at::Tensor
(of adten_t), then the at::Tensor version (ad_ten_t version)
of the virtual function should be over ridden.

If the adten_t version is over ridden, it should be evaluated using
a call_atom to a different atomic function for which the
function values are the derivative of the original atomic function.

See the previous documentation for
:ref:`atom_forward@rng_used` ,
:ref:`atom_forward@domain` , and

dom_der
*******
is the vector of the domain tensors containing the
domain directional derivative.

rng_der
*******
is the vector of the range tensors containing the
range directional derivative.

    rng_der = atom_f'(domain) * dom_der

{xrst_end atom_forward_der}
------------------------------------------------------------------------------
{xrst_begin atom_reverse_der usr}
{xrst_spell
    rng
    adten
}

Reverse Derivative Evaluation
#############################

Syntax
******
{xrst_code hpp}
    dom_der = base_atom.reverse_der(rng_used, domain, rng_der).value()
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_REVERSE_DER, END_REVERSE_DER
}
This virtual function is used if:

#.  You make a :ref:`call_atom-name` to this atomic function.
#.  The call becomes part of an AD function.
#.  The :ref:`reverse derivative<adfn_reverse_der-name>`
    of the AD function is evaluated.

If you do not over ride this virtual function its return
will have has_value() equal to false.
You should also use this case when an error occurs during
the evaluation of this derivative of the atomic function.

If rng_der[i].numel() is zero, then rng_der[i] represents a zero tensor
with the same shape as range[i] and calculations that use this value
can be skipped.

If domain in the AD function forward_der evaluation is a vector of at::Tensor
(of adten_t), then the at::Tensor version (ad_ten_t version)
of the virtual function should be over ridden.

If the adten_t version is over ridden, it should be evaluated using
a call_atom to a different atomic function for which the
function values are the derivative of the original atomic function.

See the previous documentation for
:ref:`atom_forward@rng_used` ,
:ref:`atom_forward@domain` , and

dom_der
*******
is the vector of the domain tensors containing the
domain directional derivative.

rng_der
*******
is the vector of the range tensors containing the
range directional derivative.

    dom_der = rng_der * atom_f'(domain)

{xrst_end atom_reverse_der}
------------------------------------------------------------------------------
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
    // BEGIN_CTOR
    base_atom_t(void);
    // END_CTOR
    //
    // BEGIN_NAME
    void set_name(const std::string& trace);
    const std::string& get_name(void) const;
    // END_NAME
    //
    // BEGIN_TRACE
    void set_trace(bool trace);
    bool get_trace(void) const;
    // END_TRACE
    //
    // BEGIN_DESTRUCTOR
    virtual ~base_atom_t(void) { }
    // END_DESTRUCTOR
    //
    // BEGIN_DEPEND
    virtual std::optional<sparsity_t> depend(void) const;
    // END_DEPEND
    //
    // BEGIN_FORWARD_FUN
    virtual std::optional< vector<at::Tensor> > forward(
        const vector<bool>&         rng_used,
        const vector<at::Tensor>&   domain
    ) const;
    // END_FORWARD_FUN
    //
    // BEGIN_FORWARD_DER
    virtual std::optional< vector<at::Tensor> > forward_der(
        const vector<bool>&         rng_used,
        const vector<at::Tensor>&   domain,
        const vector<at::Tensor>&   dom_der
    ) const;
    virtual std::optional< vector<adten_t> > forward_der(
        const vector<bool>&         rng_used,
        const vector<adten_t>&      domain,
        const vector<adten_t>&      dom_der
    ) const;
    // END_FORWARD_DER
    //
    // BEGIN_REVERSE_DER
    virtual std::optional< vector<at::Tensor> > reverse_der(
        const vector<bool>&        rng_used,
        const vector<at::Tensor>&  domain,
        const vector<at::Tensor>&  dom_der
    ) const;
    virtual std::optional< vector<adten_t> > reverse_der(
        const vector<bool>&        rng_used,
        const vector<adten_t>&     domain,
        const vector<adten_t>&     rng_der
    ) const;
    // END_REVERSE_DER
}; }
// END_CLASS
