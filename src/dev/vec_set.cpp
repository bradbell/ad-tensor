// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin vec_set dev}

Vector of Sets Class Specialized for Forward Mode Sparsity Calculations
#######################################################################

Properties
**********
#.  The elements of the sets are size_t values.
#.  Each set is identified by a size_t value.
#.  Once a set is created, it is not modified.
#.  A set can be created either are empty or a singleton.
#.  A set can be created as the union of other sets.

Private Member Variables
************************
{xrst_literal ,
    include/ad_tensor/dev/vec_set.hpp
    BEGIN_MEMBER_VARIABLES, END_MEMBER_VARIABLES
}

m_data
======
This vector contains all the elements and links for all the sets.

m_link
======
The length of this vector is the number of sets.
If m_link[set_id] is true, the corresponding set is a link.

m_start
=======
The length of this vector is the number of sets plus one
and the last element of m_start is equal to m_data.size().
If m_link[this_set_id] is true,

    other_set_id = m_data[ m_start[this_set_id] ]

is the set id for another set that is equal to this set.
In addition, other_set_id < this_set_id and m_link[other_set_id] is false.

If m_link[this_set_id] is false, the elements of this set are

    m_data[ m_start[this_set_id] ], ... , m_data[ m_start[this_set_id + 1] ]

Furthermore, the elements of each set are strictly increasing
with their index in data.

m_arg
=====
The length of this vector is the number of sets in the current union operation.
The i-th set in the union has set_id = m_arg[i] and  m_link[set_id] is false.

m_equal
=======
The length of this vector is the number of sets in the current union operation.
If m_equal[i] is true, all of the elements of the union are in the set
corresponding to set_id = m_arg[i].

m_next
======
Let this_set_id = m_arg[i], if m_next[i] = m_start[this_set_id + 1],
there are no more elements in this set (to be included in the union).
Otherwise, m_data[ m_next[i] ] is the next element of this set to
include in the union.

vec_set
*******
{xrst_literal ,
    BEGIN_CTOR, END_CTOR
}

data_size
*********
{xrst_literal ,
    BEGIN_N_DATA, END_N_DATA
}

new_empty
*********
{xrst_literal ,
    BEGIN_NEW_EMPTY, END_NEW_EMPTY
}
The new set id is one grater that the previous set id
(the first set id is zero).

new_singleton
*************
{xrst_literal ,
    BEGIN_NEW_SINGLETON, END_NEW_SINGLETON
}
The new set id is one grater that the previous set id
(the first set id is zero).

get_set
*******
{xrst_literal ,
    BEGIN_GET_SET, END_GET_SET
}
The elements of the set will be in increasing order.

union
*****
{xrst_literal ,
    BEGIN_UNION_SETS, END_UNION_SETS
}
The new set id is one grater that the previous set id
(the first set id is zero).
If the new set is equal to one of the sets in the union,
the new set is a link.

{xrst_end vec_set}
*/
# include<ad_tensor/dev/vec_set.hpp>

namespace ad_tensor { namespace dev { // BEGIN_AD_TENSOR_DEV

// BEGIN_CTOR
// vec_set = vec_set_t()
vec_set_t::vec_set_t(void)
: m_data(), m_link(), m_start({0}), m_arg(), m_equal(), m_next()
{ }
// END_CTOR
//
// BEGIN_N_DATA
size_t vec_set_t::data_size(void) const {
    return m_data.size();
}
// END_N_DATA
//
// BEGIN_NEW_EMPTY
// new_set_id = vec_set.new_empty()
size_t vec_set_t::new_empty(void)
{   // END_NEW_EMPTY
    //
    // new_set_id
    size_t new_set_id = m_link.size();
    //
    assert( new_set_id + 1 == m_start.size() );
    assert( m_start[new_set_id] == m_data.size() );
    //
    // m_link, m_start
    m_link.push_back(false);
    m_start.push_back( m_data.size() );
    //
    return new_set_id;
}
//
// BEGIN_NEW_SINGLETON
// new_set_id = vec_set.new_empty()
size_t vec_set_t::new_singleton(size_t element)
{   // END_NEW_SINGLETON
    //
    // new_set_id
    size_t new_set_id = m_link.size();
    //
    assert( new_set_id + 1 == m_start.size() );
    assert( m_start[new_set_id] == m_data.size() );
    //
    // m_link, m_data, m_start
    m_link.push_back(false);
    m_data.push_back(element);
    m_start.push_back( m_data.size() );
    //
    return new_set_id;
}
//
// BEGIN_GET_SET
// set = get_set(set_id)
const c10::ArrayRef<size_t> vec_set_t::get_set(size_t set_id) const
{   // END_GET_SET
#ifndef NDEBUG
    size_t n_set = m_link.size();
    assert( n_set + 1 == m_start.size() );
    assert( m_start[n_set] == m_data.size() );
#endif
    if( m_link[set_id] ) {
        set_id = m_data[ m_start[set_id] ];
    }
    assert( ! m_link[set_id] );
    //
    // set
    size_t start = m_start[set_id];
    size_t end   = m_start[set_id + 1];
    assert( end <= m_data.size() );
    c10::ArrayRef<size_t> set( m_data.data() + start, m_data.data() + end );
    return set;
}
//
// BEGIN_UNION_SETS
size_t vec_set_t::union_sets( const c10::ArrayRef<size_t>& sets)
{   // END_UNION_SETS
    //
    // m_arg, m_equal, m_next
    m_arg.resize(0);
    m_equal.resize(0);
    m_next.resize(0);
    for(size_t i = 0; i < sets.size(); ++i) {
        //
        // set_id
        size_t set_id = sets[i];
        if( m_link[set_id] ) {
            set_id = m_data[ m_start[set_id] ];
        }
        assert( ! m_link[set_id] );
        //
        bool empty = m_start[set_id] == m_start[set_id + 1];
        if( ! empty ) {
            m_arg.push_back(set_id);
            m_equal.push_back(true);
            m_next.push_back( m_start[set_id] );
        }
    }
    if( m_arg.size() == 0 ) {
        return new_empty();
    }
    //
    // i_min
    size_t i_min = 0;
    for(size_t i = 1; i < m_arg.size(); ++i) {
        if( m_data[ m_next[i] ]  < m_data[ m_next[i_min] ] ) {
            i_min = i;
        }
    }
    //
    // new_set_id
    size_t new_set_id = m_link.size();
    assert( m_start[ new_set_id ] == m_data.size() );
    //
    // m_data, m_equal
    while( i_min < m_arg.size() ) {
        //
        // element, m_data
        size_t element = m_data[ m_next[i_min] ];
        m_data.push_back(element);
        //
        // m_next, m_equal
        for(size_t i = 0; i < m_arg.size(); ++i) {
            size_t set_id = m_arg[i];
            if( m_next[i] == m_start[set_id + 1] ) {
                m_equal[i] = false;
            } else {
                assert( element <= m_data[ m_next[i] ] );
                if( element == m_data[ m_next[i] ] ) {
                    ++m_next[i];
                } else {
                    m_equal[i] = false;
                }
            }
        }
        //
        // i_min
        i_min = m_arg.size();
        for(size_t i = 0; i < m_arg.size(); ++i) {
            size_t set_id = m_arg[i];
            if( m_next[i] < m_start[set_id + 1] ) {
                if( i_min == m_arg.size() ) {
                    i_min = i;
                } else if( m_data[ m_next[i] ]  < m_data[ m_next[i_min] ] ) {
                    i_min = i;
                }
            }
        }
    }
    //
    // m_data, m_link, m_start
    for(size_t i = 0; i < m_arg.size(); ++i) {
        if( m_equal[i] ) {
            //
            size_t equal_set_id = m_arg[i];
            assert( ! m_link[equal_set_id] );
            m_data.resize( m_start[ new_set_id ] );
            m_link.push_back(true);
            m_data.push_back( equal_set_id );
            m_start.push_back( m_data.size() );
            return new_set_id;
        }
    }
    //
    // m_link, m_start
    m_link.push_back(false);
    m_start.push_back( m_data.size() );
    return new_set_id;
}

} } // END_AD_TRENSOR_DEV
