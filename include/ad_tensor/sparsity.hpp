#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin sparsity usr}

Sparsity Patterns
#################

{xrst_end sparsity}
*/
#include<array>
#include<ad_tensor/vector.hpp>

namespace ad_tensor {
    class sparsity_t {
    private:
        vector< std::array<size_t, 2> > m_sparsity;
    public:
        size_t size(void);
        void sort(void);
        void push_back(const std::array<size_t, 2>& pair );
        const std::array<size_t, 2>& operator[](size_t index) const;
    };
}
