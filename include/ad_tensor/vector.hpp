#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <vector>
/*
{xrst_begin vector usr}
{xrst_spell
    std
}

The AD Tensor Vector Class
##########################
This a light weight wrapper for ``std::vector`` that only modifies
the indexing operator when NDEBUG is not defined.
To be specific, the indexing operator [] is mapped
to std::vector::at so that indices are checked to be in bounds.
In all other aspects, it is exactly like std::vector.

Prototype
*********
{xrst_literal ,
    BEGIN_TOP, END_TOP
    BEGIN_BOTTOM, END_BOTTOM
}

{xrst_end vector}
*/
// BEGIN_TOP
namespace ad_tensor {
    template <class T> class vector : public std::vector<T>
    {
    public:
        typedef typename std::vector<T>::size_type       size_type;
        typedef typename std::vector<T>::value_type      value_type;
        typedef typename std::vector<T>::reference       reference;
        typedef typename std::vector<T>::const_reference const_reference;
        // ...
        // constructors and assignment member functions here
        // ...
// END_TOP
        //
        // constructors
        vector(void)
        : std::vector<T>()
        { }
        vector(size_type count)
        : std::vector<T>(count)
        { }
        vector(size_type count, const T& value)
        : std::vector<T>(count, value)
        { }
        template <class InputIt> vector(InputIt first, InputIt last)
        : std::vector<T>(first, last)
        { }
        vector(const vector& other)
        : std::vector<T>(other)
        { }
        vector(const vector&& other)
        : std::vector<T>(other)
        { }
        vector(std::initializer_list<T> init)
        : std::vector<T>(init)
        { }
        //
        // assignments
        vector& operator=(const vector& other) {
            std::vector<T>::operator=(other);
            return *this;
        }
        vector& operator=(const vector&& other) {
            std::vector<T>::operator=(other);
            return *this;
        }
        vector& operator=( std::initializer_list<value_type> ilist )
        {   std::vector<T>::operator=(ilist);
            return *this;
        }
        //
// BEGIN_BOTTOM
# ifndef NDEBUG
        reference operator[](size_type pos)
        {   return this->at(pos); }
        const_reference operator[](size_type pos) const
        {   return this->at(pos); }
# endif
// END_BOTTOM
    };
}
