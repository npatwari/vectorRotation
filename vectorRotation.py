#! /usr/bin/env python

#
# LICENSE:
# Copyright (C) 2017  Neal Patwari
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
# 
# Author: Neal Patwari, neal.patwari@gmail.com
#
#
# Version History:
#
# Version 1.0:  Initial Release.  8 Feb 2017
#

import cmath
import numpy as np

# PURPOSE: Rotate vector y by multiplying it with scalar exp(j\theta) to best
#   match the other input vector x.  That is, find 
#
#    argmin_\theta | x - exp(j\theta) y |**2 
#
#   and then compute (and return) the rotated vector exp(j\theta) y.
#   Note \theta is a scalar phase, thus exp(j\theta) is a complex scalar value.
#   Thus we are finding one single angle \theta to rotate every element of y
#   so that it is most similar to x in a least-squares sense.
#
#   The solution to the minimization is that \theta = - phase(x^H y).  That is,
#   we set \theta to (-1)*the phase of the inner product of 
#   (the hermitian of x) and (y).
#
# INPUTS: numpy ndarrays x and y
#
# OUTPUTS: numpy ndarray r = exp(j\theta) y; and \theta
#
def rotateCVectorsToMatch(y, x):
    inner_product = np.dot(x.conj().T, y)            # x^H * y
    theta         = -1.0*cmath.phase(inner_product)  # \theta
    ejt           = cmath.exp(1J*theta)              # exp(j\theta)
    r             = ejt * y
    return (r, theta)