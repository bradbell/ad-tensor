// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adfn_source_gen usr}
{xrst_spell
    std
    ptr
    vptr
    dll
}

Generate C++ Source Code For this AD Function
#############################################

Under Construction
******************
The documentation below is just a plan.
Its implementation is under construction.

Syntax
******
{xrst_code cpp}
    source_code = adfn.source_gen()
{xrst_code}

adfn
****
We use adfn for this AD function object.

source_code
***********
Is C++ source code for the function range = adfn(dom_var, dom_par) where:
{xrst_code cpp}
    par_all = adfn.forward_par(dom_par);
    var_all = adfn.forward_var(dom_var, par_all);
    y       = adfn.get_range(var_all, par_all);
{xrst_code}
The vector y above can contain derivatives if adfn
was recording using AD Tensors and derivatives of another AD function.
The C interface to the source code function is defined below.
(A C interface is used to make dll linking simpler.)

C Interface
***********

Syntax
======
{xrst_code cpp}
    range_vptr = adfn_name(dom_par_vptr, dom_var_ptr, message_vptr)
{xrst_code}

vector_tensor_t
===============
{xrst_literal ,
    BEGIN_VECTOR_TENSOR_T, END_VECTOR_TENSOR_T
}

Prototype
=========
{xrst_literal ,
    BEGIN_ADFN_NAME , END_ADFN_NAME
}
{xrst_code cpp}
}
{xrst_code}

adfn_name
---------
is the name of this adfn; see :ref:`stop_recording-name` .

dom_par_vptr
------------
is the vector of domain parameter tensors as the following
{xrst_code cpp}
    dom_par_vptr = static_cast<void*>( vector_tensor_t* );
{xrst_code}

dom_var_vptr
------------
is the vector of domain variable tensors as the following
{xrst_code cpp}
    dom_var_vptr = static_cast<void*>( vector_tensor_t* );
{xrst_code}

message_vptr
------------
On input \*message_ptr must be the empty string.
If it is empty on return, no error or warnings occurred.
Otherwise, it is a warning or error message.
warning message:
{xrst_code cpp}
    std::string* message_ptr = static_cast<std::string>(message_vptr);
{xrst_code}

range_vptr
----------
If a fatal error occurs, the return will be a null pointer.
Otherwise, it is the vector of range results as the following
{xrst_code cpp}
    range_vptr = static_cast<void*>( vector_tensor_t* );
{xrst_code}
This vector is allocated by adfn_name and must be deleted by the
calling program.

{xrst_end adfn_source_gen}
*/
#include <torch/torch.h>
#include <gtest/gtest.h>
#include <ad_tensor/to_string.hpp>
#include <ad_tensor/vector.hpp>
//
namespace { // BEGIN_EMPTY_NAMESPACE
//
// BEGIN_VECTOR_TENSOR_T
typedef ad_tensor::vector< std::unique_ptr<at::Tensor> > vector_tensor_t;
// END_VECTOR_TENSOR_T
//
// BEGIN_ADFN_NAME
extern "C" { void* adfn_name(
    void*  dom_par_vptr,
    void*  dom_var_vptr,
    void*   message_vptr
    ) // END_ADFN_NAME
    {
        //
        // dom_par_ptr, dom_var_ptr, message_ptr
        auto dom_par_ptr = static_cast<vector_tensor_t*>( dom_par_vptr );
        auto dom_var_ptr = static_cast<vector_tensor_t*>( dom_var_vptr );
        auto message_ptr = static_cast<std::string*>( message_vptr );
        //
        // dom_par, dom_var, message
        vector_tensor_t& dom_par  = *dom_par_ptr;
        vector_tensor_t& dom_var  = *dom_var_ptr;
        std::string&     message  = *message_ptr;
        assert( message == "" );
        //
        //
        // dom_par
        EXPECT_EQ( dom_par.size(), 1);
        EXPECT_TRUE(torch::tensor( {1, 2} ).equal( *(dom_par[0]) ));
        //
        // dom_var
        EXPECT_EQ( dom_var.size(), 1);
        EXPECT_TRUE(torch::tensor( {{3, 4}, {5, 6}} ).equal( *(dom_var[0]) ));
        //
        // message
        message = "example warning";
        //
        // range
        std::unique_ptr<at::Tensor> range_0 = std::make_unique<at::Tensor>();
        *range_0 = torch::tensor( {7, 8, 9} );
        vector_tensor_t* range_ptr = new vector_tensor_t;
        (*range_ptr).push_back( std::move(range_0) );
        return static_cast<void*>( range_ptr );
} }

} // END_EMPTY_NAMESPACE

TEST(tests_adten, source_gen)  {

    using ad_tensor::vector;
    //
    // dom_par
    std::unique_ptr<at::Tensor> dom_par_0 = std::make_unique<at::Tensor>();
    *dom_par_0 = torch::tensor( {1, 2} );
    vector_tensor_t dom_par;
    dom_par.push_back( std::move(dom_par_0) );
    //
    // dom_var
    std::unique_ptr<at::Tensor> dom_var_0 = std::make_unique<at::Tensor>();
    *dom_var_0 = torch::tensor( { {3, 4}, {5, 6} } );
    vector_tensor_t dom_var;
    dom_var.push_back( std::move(dom_var_0) );
    //
    // message
    std::string message = "";
    //
    // range, message
    void* range_vptr = adfn_name(
        static_cast<void*>( &dom_par )      ,
        static_cast<void*>( &dom_var )      ,
        static_cast<void*>( &message )
    );
    vector_tensor_t* range_ptr = static_cast<vector_tensor_t*>(range_vptr);
    vector_tensor_t& range     = *range_ptr;
    //
    // message
    EXPECT_TRUE( message == "example warning" );
    //
    // range
    EXPECT_EQ( range.size(), 1);
    EXPECT_TRUE( torch::tensor({7, 8, 9}).equal( *(range[0]) ) );
}
