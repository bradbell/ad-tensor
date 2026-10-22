// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adfn usr}

The AD Function Class
#####################

Default Constructor
*******************
This creates an empty AD function; i.e.,
one that does not have any operations:
{xrst_literal ,
    BEGIN_DEFAULT_CTOR, END_DEFAULT_CTOR
}

is_empty
********
Returns true if there are no operations in this AD function:
{xrst_literal ,
    BEGIN_IS_EMPTY, END_IS_EMPTY
}

get_name
********
get_name returns the name chosen by :ref`stop_recording-name` :
{xrst_literal ,
    BEGIN_GET_NAME, END_GET_NAME
}

trace
*****
{xrst_literal ,
    BEGIN_GET_TRACE, END_GET_TRACE
    BEGIN_SET_TRACE, END_SET_TRACE
}
get_trace returns the previous value chosen by set_trace.
The default value, when trace has not been set, is false.

n_range
*******
{xrst_literal ,
    BEGIN_N_RANGE, END_N_RANGE
}
Returns the number of tensors in the range space for this AD function.

n_con
*****
{xrst_literal ,
    BEGIN_N_CON, END_N_CON
}
Returns the number of constants in the AD function.
See the discussion about the :ref:`start_recording@First Constant`.

n_par
*****
{xrst_literal ,
    BEGIN_N_PAR, END_N_PAR
}
Returns the number of parameters in the AD function; i.e.,
the length of the vector :ref:`adfn_forward_par@par_all` .

n_var
*****
{xrst_literal ,
    BEGIN_N_VAR, END_N_VAR
}
Returns the number of variables in the AD function; i.e.,
the length of the vector :ref:`adfn_forward_var@var_all` .

print_con
*********
{xrst_literal ,
    BEGIN_PRINT_CON, END_PRINT_CON
}
print the constants stored in an AD function object.
This may be useful for understanding the trace output.

Other Public Members
********************
{xrst_toc_table
    src/adfn/forward_dep.cpp
    src/adfn/forward_par.cpp
    src/adfn/forward_var.cpp
    src/adfn/get_range.cpp
    src/adfn/forward_der.cpp
    src/adfn/reverse_der.cpp
    src/adfn/optimize.cpp
    src/adfn/src_gen.cpp
}


{xrst_end adfn}
------------------------------------------------------------------------------
{xrst_begin adfn_dev dev}
{xrst_spell
    rng
}

The AD Function Class Private Members
#####################################
{xrst_literal ,
    include/ad_tensor/adfn.hpp
    BEGIN_MEMBER_DATA, END_MEMBER_DATA
}

m_par
*****
is the acyclic graph for the dependent parameters.

m_var
*****
is the acyclic graph for the dependent variables.

m_con_vec
*********
is the vector of constant tensors.

m_rng_index
***********
is the constant, parameter, or variable index for each of the range tensors.

m_rng_adtype
************
is the adtype for each of the range tensors.

m_rng_shapes
************
is the shape for each of the range tensors.

{xrst_end adfn_dev}
------------------------------------------------------------------------------
*/
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/to_string.hpp>
namespace ad_tensor {
    //
    // BEGIN_DEFAULT_CTOR
    adfn_t::adfn_t(void)
    // END_DEFAULT_CTOR
    : m_con_vec()
    , m_par()
    , m_var()
    , m_rng_index()
    , m_rng_adtype()
    , m_rng_shapes()
    , m_name()
    , m_trace(false)
    { }
    //
    // BEGIN_IS_EMPTY
    bool adfn_t::is_empty(void) const
    // END_IS_EMPTY
    { return
        m_par.is_empty() &&
        m_var.is_empty() &&
        m_con_vec.empty() &&
        m_rng_index.empty() &&
        m_rng_adtype.empty() &&
        m_rng_shapes.empty() &&
        m_name.empty()
        ;
    }
    //
    // BEGIN_N_RANGE
    size_t adfn_t::n_range(void) const
    {   // END_N_RANGE
        return m_rng_index.size();
    }
    //
    // BEGIN_N_CON
    size_t adfn_t::n_con(void) const
    {   // END_N_CON
        return m_con_vec.size();
    }
    //
    // BEGIN_N_PAR
    size_t adfn_t::n_par(void) const
    {   // END_N_PAR
        return m_par.m_op_seq.size();
    }
    //
    // BEGIN_N_VAR
    size_t adfn_t::n_var(void) const
    {   // END_N_VAR
        return m_var.m_op_seq.size();
    }
    //
    // BEGIN_GET_NAME
    const std::string& adfn_t::get_name(void) const
    {   // END_GET_NAME
        return m_name;
    }
    //
    // BEGIN_GET_TRACE
    bool adfn_t::get_trace(void) const
    {   // END_GET_TRACE
        return m_trace;
    }
    //
    // BEGIN_SET_TRACE
    void adfn_t::set_trace(bool trace)
    {   // END_SET_TRACE
        m_trace = trace;
    }
    //
    // BEGIN_PRINT_CON
    void adfn_t::print_con(void) const
    {   // END_PRINT_CON
        std::cout << "Begin " + get_name() + ".print_con\n";
        std::cout << to_string( m_con_vec );
        std::cout << "End " + get_name() + ".print_con\n";
    }
}
