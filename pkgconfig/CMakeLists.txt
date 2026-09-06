# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
#
# ad_tensor.pc
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/ad_tensor.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ad_tensor.pc"
    @ONLY
)
#
# install
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ad_tensor.pc"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
)
