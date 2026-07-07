// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin_parent atom usr}

Atomic Functions
################
Atomic functions have the same benefit as :ref:`chkpnt-title`
but with the added flexibility and difficulty that the user
defines the derivative computations; see :ref:`base_atom-name` .

{xrst_toc_table after
    include/ad_tensor/base_atom.hpp
}

{xrst_end atom}
-----------------------------------------------------------------------------
{xrst_begin atom_examples usr}

Checkpoint Examples
###################
{xrst_toc_table
    examples/atom/get_started.cpp
}

{xrst_end atom_examples}
------------------------------------------------------------------------------
{xrst_begin make_atom usr}
{xrst_spell
    ptr
}

Create An Atomic Function
#########################

Syntax
******
{xrst_code cpp}
atom_id = make_atom(base_atom_ptr)
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_MAKE_ATOM, END_MAKE_ATOM
}

base_atom_ptr
*************
On input, this a pointer the :ref:`base_atom-name` object.
The class for this object should be derived from base_atom_t
and should fill in the virtual functions that are needed.
Upon return, this pointer has been replaced by a pointer
to a default constructed base class object.

atom_id
*******
is the identifier for this atomic function.


{xrst_end make_atom}
-----------------------------------------------------------------------------
{xrst_begin call_atom usr}

Calling An Atomic Function
##########################
{xrst_spell
    adomain
    arange
}

Syntax
******
{xrst_code cpp}
arange = call_atom(atom_id, call_info, adomain)
{xrst_code}

Prototype
*********
{xrst_literal ,
    // BEGIN_CALL_ATOM, END_CALL_ATOM
}

atom_id
*******
is the atomic function identifier; see :ref:`make_atom@atom_id` .

call_info
*********
is the call information that gets passed from the atomic function call
through to its virtual functions; see :ref:`atom_forward@call_info` .

adomain
*******
is the AD tensor version of the domain for this atomic function call.

arange
******
is the AD tensor version of the range for this atomic function call.

{xrst_end call_atom}
*/
#include <ad_tensor/atom.hpp>
#include <ad_tensor/dev/atom.hpp>
namespace ad_tensor {
    //
    // BEGIN_MAKE_ATOM
    size_t make_atom(std::unique_ptr<base_atom_t>& base_atom_ptr)
    {   // END_MAKE_ATOM
        //
        dev::atom_global_t& global = dev::atom_global_t::singleton();
        size_t atom_id             = global.store(base_atom_ptr);
        return atom_id;
    }
    //
    // BEGIN_CALL_ATOM
    vector<adten_t> call_atom(
    size_t                 atom_id   ,
    size_t                 call_info ,
    const vector<adten_t>& adomain   )
    {   // END_CALL_ATOM
        return adten_t::call_atom(atom_id, call_info, adomain);
    }
}
