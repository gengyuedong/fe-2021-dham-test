# Cause division to always mean floating point division.
from __future__ import division
import numpy as np
from .reference_elements import ReferenceInterval, ReferenceTriangle
np.seterr(invalid='ignore', divide='ignore')


def lagrange_points(cell, degree):
    """Construct the locations of the equispaced Lagrange nodes on cell.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct nodes.

    :returns: a rank 2 :class:`~numpy.array` whose rows are the
        coordinates of the nodes.

    The implementation of this function is left as an :ref:`exercise
    <ex-lagrange-points>`.

    """

    # Create an empty list to store node coordinates
    points = []
    if cell.dim == 1:
        points = np.linspace(0,1, degree + 1).reshape(-1,1)

    else:
    # Loop through all valid barycentric coordinates
        for i in range(degree+1):
            for j in range(degree-i+1):
                x = i / degree
                y = j / degree

                points.append(np.array([x,y]))
        points = np.array(points)
    # print(points)
    
    return points
    raise NotImplementedError


def vandermonde_matrix(cell, degree, points, grad=False):
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree on the cell provided.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct the matrix.
    :param points: a list of coordinate tuples corresponding to the points.
    :param grad: whether to evaluate the Vandermonde matrix or its gradient.

    :returns: the generalised :ref:`Vandermonde matrix <sec-vandermonde>`

    The implementation of this function is left as an :ref:`exercise
    <ex-vandermonde>`.
    """
    points = np.array(points)
    if grad == False:
        
        # print(points)
        # print(degree)
        if cell.dim == 1:
            result = np.zeros((len(points),degree+1))

            for j in range(degree+1):
                result[:,j] = points[:,0] ** j
            return result 
        
        elif cell.dim == 2:
            num_terms = (degree + 1) * (degree + 2) // 2 
            result = np.zeros((len(points),num_terms))
            # print(points, result)
            col = 0
            for n in range(degree + 1):
                for m in range(n + 1):
                    result[:,col] = (points[:,0] ** (n-m)) * points[:,1]**m
                    col += 1

            return result
        else:
            raise NotImplementedError
    else:
        if cell.dim == 1:
            result = np.zeros((len(points),degree+1,1))
            for j in range(degree + 1):
                print(1)
                if j == 0:
                    result[:, j, 0] = 0  # Derivative of constant term is 0
                else:
                    result[:, j, 0] = j * points[:, 0] ** (j - 1)
            return result
        elif cell.dim == 2:
            num_terms = (degree + 1) * (degree + 2) // 2 
            result = np.zeros((len(points),num_terms,2))
            col = 0 
            for n in range(degree + 1):
                for m in range(n+1):
                    # print(1)
                    result[:,col,0] = (n - m) * (points[:, 0] ** (n - m - 1)) * (points[:, 1] ** m) if (n - m) > 0 else 0
                    result[:,col,1] = m * (points[:, 0] ** (n - m)) * (points[:, 1] ** (m - 1)) if m > 0 else 0
                    col+=1
            return result
        else:
            raise NotImplementedError
class FiniteElement(object):
    def __init__(self, cell, degree, nodes, entity_nodes=None):
        """A finite element defined over cell.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        :param nodes: a list of coordinate tuples corresponding to
            the nodes of the element.
        :param entity_nodes: a dictionary of dictionaries such that
            entity_nodes[d][i] is the list of nodes associated with entity `(d, i)`.

        Most of the implementation of this class is left as exercises.
        """

        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = cell
        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = degree
        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = nodes
        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)`.
        self.entity_nodes = entity_nodes

        if entity_nodes:
            #: ``nodes_per_entity[d]`` is the number of entities
            #: associated with an entity of dimension d.
            self.nodes_per_entity = np.array([len(entity_nodes[d][0])
                                              for d in range(cell.dim+1)])

        # Replace this exception with some code which sets
 
        V = vandermonde_matrix(self.cell,self.degree,nodes)
        self.basis_coefs = np.linalg.inv(V)
        # to an array of polynomial coefficients defining the basis functions.
        # raise NotImplementedError

        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``False`` and (points, nodes, dim) if ``grad``
            is ``True``.

        The implementation of this method is left as an :ref:`exercise
        <ex-tabulate>`.

        """
        if grad == False:
            V = vandermonde_matrix(self.cell,self.degree,points)
            return V @ self.basis_coefs
        else:
            V = vandermonde_matrix(self.cell,self.degree,points,grad=True)
            return np.einsum('ijk,jl->ilk', V, self.basis_coefs)
        
        raise NotImplementedError

    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a scalar value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.

        The implementation of this method is left as an :ref:`exercise
        <ex-interpolate>`.

        """
        # Check the dimensionality of the element
        if self.nodes.shape[1] == 1:  # 1D case
            fn_vals = np.zeros(len(self.nodes))  # Initialize 1D array for function values
            
            # Loop over the 1D nodes and evaluate fn at each node
            for i, node in enumerate(self.nodes):
                fn_vals[i] = fn(node[0])  # Node is a scalar in 1D (use node[0] to get the scalar)
            return fn_vals
        elif self.nodes.shape[1] == 2:  # 2D case
            fn_vals = np.zeros(len(self.nodes))  # Initialize 1D array for function values
            
            # Loop over the 2D nodes and evaluate fn at each node
            for i, node in enumerate(self.nodes):
                fn_vals[i] = fn(node)  # Node is a vector in 2D (e.g., [x, y])
            return fn_vals
        else:
            raise NotImplementedError("Interpolation for higher dimensions is not yet implemented.")
        #raise NotImplementedError

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.cell,
                               self.degree)


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """An equispaced Lagrange finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.

        The implementation of this class is left as an :ref:`exercise
        <ex-lagrange-element>`.
        """

        # raise NotImplementedError
        # Use lagrange_points to obtain the set of nodes.  Once you
        # have obtained nodes, the following line will call the
        # __init__ method on the FiniteElement class to set up the
        # basis coefficients.
        super(LagrangeElement, self).__init__(cell, degree, lagrange_points(cell,degree))
