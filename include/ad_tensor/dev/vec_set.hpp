#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
#include <ad_tensor/vector.hpp>

namespace ad_tensor { namespace dev {
    class vec_set_t {
    private:
        // BEGIN_MEMBER_VARIABLES
        vector<size_t> m_data;
        vector<bool>   m_link;
        vector<size_t> m_start;
        vector<size_t> m_arg;
        vector<bool>   m_equal;
        vector<size_t> m_next;
        // END_MEMBER_VARIABLES
    public:
        vec_set_t(void);
        size_t data_size(void) const;
        size_t new_empty(void);
        size_t new_singleton(size_t element);
        const c10::ArrayRef<size_t> get_set(size_t set_id) const;
        size_t union_sets( const c10::ArrayRef<size_t>& sets);
    };
} }
