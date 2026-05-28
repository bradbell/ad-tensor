#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <vector>
//
namespace ad_tensor {
# ifdef NDEBUG
        using std::vector;
# else
    template <class T> class vector : public std::vector<T> {
        typedef std::vector<T>::size_type       size_type;
        typedef std::vector<T>::value_type      value_type;
        typedef std::vector<T>::reference       reference;
        typedef std::vector<T>::const_reference const_reference;
    public:
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
        // indexing
        reference operator[](size_type pos)
        {   return this->at(pos); }
        const_reference operator[](size_type pos) const
        {   return this->at(pos); }
    };
# endif
}
