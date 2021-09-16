from dolfin import *
from spaces import *
import sys
import os

import math
import numpy as np
import scipy.sparse as sps
import scipy.io as sio
import scipy.sparse.linalg as la
import time
import matplotlib.pyplot as plt

######################################### Cartesian Homogeneous 2 Components ####################################
class LIAproblem2D():
   def __init__(self, Re, respath):
      
      self.Re = Re
      self.Pr = 1.0
      self.respath = respath
      os.makedirs(respath,exist_ok=True) # create dir if necessary
     
      self.X = X   #from spaces.py
      self.Urho = Urho
      self.Uv = Uv
      self.Up = Up
      print('Number of degrees of freedom = '+str(self.X.dim()))
    ###############################################################################
    ## Define Boundaries
      def cylinder(x, on_boundary):  #symétrie de l'écoulement stationnaire
         return (x[0]**2+x[1]**2)**0.5 < 0.5001 and on_boundary 
      def inlet(x, on_boundary):       
         return x[0] < (-9.9999) and on_boundary 
      def outlet(x, on_boundary):         
         return x[0] > (29.9999) and on_boundary 
      def upperandlower(x, on_boundary):
         return np.abs(x[1]) > 9.9999 and on_boundary 
      # define boundary conditions for baseflow & DNS
      bcs_cyl0=DirichletBC(X.sub(0), 1/1.3, cylinder)
      bcs_cyl1=DirichletBC(X.sub(1), (0,0), cylinder)
      bcs_inflow0 = DirichletBC(X.sub(0), 1, inlet)
      bcs_inflow1 = DirichletBC(X.sub(1), (1,0), inlet)
      bcs_upperandlower0=DirichletBC(X.sub(0), 1, upperandlower)
      bcs_upperandlower1=DirichletBC(X.sub(1).sub(1), 0, upperandlower)
      self.bcs = [bcs_cyl0,bcs_cyl1,bcs_inflow0,bcs_inflow1,bcs_upperandlower0,bcs_upperandlower1]
      # define boundary conditions for perturbations
      bcp_cyl0=DirichletBC(X.sub(0), 0, cylinder)
      bcp_cyl1=DirichletBC(X.sub(1), (0,0), cylinder)
      bcp_inflow0 = DirichletBC(X.sub(0), 0, inlet)
      bcp_inflow1 = DirichletBC(X.sub(1), (0,0), inlet)
      bcp_upperandlower0=DirichletBC(X.sub(0), 0, upperandlower)
      bcp_upperandlower1=DirichletBC(X.sub(1).sub(1), 0, upperandlower)
      self.bcp = [bcp_cyl0,bcp_cyl1,bcp_inflow0,bcp_inflow1,bcp_upperandlower0,bcp_upperandlower1]

# solves steady state equations
   def steady_state_trueDensity(self):
      rup = Function(self.X) # function object representing the solution
      Uinit = Expression(("1", "1", "0", "0"),degree=1)
      rup = interpolate(Uinit, self.X)
      
      dup = TrialFunction(self.X)
      vp = TestFunction(self.X) 
      
      Re = Constant(self.Re)
      Pr = Constant(self.Pr)
      # define the nonlinear problem
      rho= rup[0]
      u  = as_vector((rup[1],rup[2]))   # velocity
      p  = rup[3]                      # pressure
      s  = vp[0]
      v  = as_vector((vp[1],vp[2]))   # velocity
      q  = vp[3]                      # pressure
      F   =  inner(grad(u)*u*rho, v)*dx      \
          + 1/Re*inner(grad(u), grad(v))*dx \
          - div(v)*p*dx + q*rho*div(u)*dx - 1/(Re*Pr)*dot(rho**(-2)*grad(rho), grad(rho*q))*dx \
          + dot(grad(rho),u)*s*dx \
          + 1/(Re*Pr)*dot(rho**(-2)*grad(rho), grad(rho*s))*dx
      # define its Jacobian
      dF  = derivative(F, rup, dup) 
      problem = NonlinearVariationalProblem(F, rup, self.bcs, dF)
      # solve the problem with Newton
      solver  = NonlinearVariationalSolver(problem)
      solver.parameters['newton_solver']['relative_tolerance']  = 1e-14
      solver.parameters['newton_solver']['absolute_tolerance']  = 1e-14
      solver.parameters['newton_solver']['maximum_iterations']  = 10
      solver.solve()
       # write to file
      File(self.respath+"baseflow.xml") << rup.vector()
      rho,u,p = rup.split()
      rho.rename("rho","density"); u.rename("v","velocity");  p.rename("p","pressure");
      print("Saving vtk files baseflow_u.pvd, baseflow_p.pvd ")
      File(self.respath+"baseflow_rho.pvd") << rho
      File(self.respath+"baseflow_u.pvd") << u
      File(self.respath+"baseflow_p.pvd") << p
      
      cs=plot(rho)
      plt.show()
      
# solves steady state equations
   def steady_state_simplifiedDensity(self):
      rup = Function(self.X) # function object representing the solution
      Uinit = Expression(("1", "1", "0", "0"),degree=1)
      rup = interpolate(Uinit, self.X)
      
      dup = TrialFunction(self.X)
      vp = TestFunction(self.X) 
      
      Re = Constant(self.Re)
      Pr = Constant(self.Pr)
      # define the nonlinear problem
      rho= rup[0]
      u  = as_vector((rup[1],rup[2]))   # velocity
      p  = rup[3]                      # pressure
      s  = vp[0]
      v  = as_vector((vp[1],vp[2]))   # velocity
      q  = vp[3]                      # pressure
      F   =  inner(grad(u)*u*rho, v)*dx      \
          + 1/Re*inner(grad(u), grad(v))*dx \
          - div(v)*p*dx + q*rho*div(u)*dx - 1/(Re*Pr)*dot(grad(rho), grad(q))*dx \
          + dot(grad(rho),u)*s*dx \
          + 1/(Re*Pr)*dot(grad(rho), grad(s))*dx
      # define its Jacobian
      dF  = derivative(F, rup, dup) 
      problem = NonlinearVariationalProblem(F, rup, self.bcs, dF)
      # solve the problem with Newton
      solver  = NonlinearVariationalSolver(problem)
      solver.parameters['newton_solver']['relative_tolerance']  = 1e-14
      solver.parameters['newton_solver']['absolute_tolerance']  = 1e-14
      solver.parameters['newton_solver']['maximum_iterations']  = 10
      solver.solve()
       # write to file
      File(self.respath+"baseflow.xml") << rup.vector()
      rho,u,p = rup.split()
      rho.rename("rho","density"); u.rename("v","velocity");  p.rename("p","pressure");
      print("Saving vtk files baseflow_u.pvd, baseflow_p.pvd ")
      File(self.respath+"baseflow_rho.pvd") << rho
      File(self.respath+"baseflow_u.pvd") << u
      File(self.respath+"baseflow_p.pvd") << p
      
      cs=plot(rho)
      plt.show()
      
           
#---------------------------------------------------------------------------------------#      
   # Returns dof indices which are free
   # freeinds = free indices of velocity, temperature, pressure
   # pinds    = free indices of pressure
   def get_indices(self):
      # Collect all dirichlet boundary dof indices
      bcinds = []
      for b in self.bcp:
         bcdict = b.get_boundary_values()
         bcinds.extend(bcdict.keys())

      # total number of dofs
      N = self.X.dim()
      
      # indices of free nodes
      freeinds = np.setdiff1d(range(N),bcinds,assume_unique=True).astype(np.int32)

      # pressure indices
      pinds = self.X.sub(2).dofmap().dofs()

      return freeinds, pinds
#----------------------------------------------------------------------------------------
   # Compute k eigenvalues/vectors   
   def eigenvalues(self, sigma=0, k=0):
      parameters['linear_algebra_backend'] = 'Eigen'
       
      # load baseflow
      rup = Function(self.X)
      File(self.respath+"baseflow.xml") >> rup.vector()
      
      Re = Constant(self.Re)
      Pr = Constant(self.Pr)
      
      # define the nonlinear problem
      drup = TrialFunction(self.X)
      rho,u,p = TrialFunctions(self.X)
      s,v,q = TestFunctions(self.X)
      vp = TestFunction(self.X) 
      
      # define RHS matrix
      Ma = assemble(inner(u,v)*dx + rho*s*dx)
      # Convert to sparse format
      rows, cols, values = as_backend_type(Ma).data()
      Ma = sps.csr_matrix((values, cols, rows))
      
      rho= rup[0]
      u  = as_vector((rup[1],rup[2]))   # velocity
      p  = rup[3]                      # pressure
      s  = vp[0]
      v  = as_vector((vp[1],vp[2]))   # velocity
      q  = vp[3]                      # pressure
      F   =  -( inner(grad(u)*u*rho, v)*dx      \
          + 1/Re*inner(grad(u), grad(v))*dx \
          - div(v)*p*dx + q*rho*div(u)*dx - 1/(Re*Pr)*dot(rho**(-2)*grad(rho), grad(rho*q))*dx \
          + dot(grad(rho),u)*s*dx \
          + 1/(Re*Pr)*dot(rho**(-2)*grad(rho), grad(rho*s))*dx)
      # define its Jacobian
      Aform  = derivative(F, rup, drup) 
      Aa = assemble(Aform)
      # Convert to sparse format
      rows, cols, values = as_backend_type(Aa).data()
      Aa = sps.csr_matrix((values, cols, rows))

      # remove Dirichlet points from the system
      freeinds,pinds = self.get_indices() 
      M = Ma[freeinds,:][:,freeinds]
      A = Aa[freeinds,:][:,freeinds]

      if k>0:
        # Compute eigenvalues/vectors of (A,M)
        print("Computing eigenvalues/vectors ...")
        ncv = np.max([10,2*k]) # number of Krylov vectors
        vals, vecs = la.eigs(A, k=k, M=M, sigma=sigma,  ncv=ncv, maxiter=40, tol=10e-10) 
        
        file = open(self.respath+"evals.dat","w")
        for val in vals:
            print(np.real(val), np.imag(val))
            file.write("%s\n" % val)
        file.close()
        
        # only writing real parts of eigenvectors to file
        ua = Function(self.X)
        for i in range(0,k):
            ua.vector()[freeinds] = vecs[:,i]
            File(self.respath+"evec"+str(i+1)+".xml") << ua.vector()
            rho,u,p  = ua.split()
            File(self.respath+"evec_rho_"+str(i+1)+".pvd") << rho
            File(self.respath+"evec_u_"+str(i+1)+".pvd") << u
            File(self.respath+"evec_p_"+str(i+1)+".pvd") << p


#----------------------------------------------------------------------------------------

   def lowmachTimestepper_CN_Newton_trueDensity(self, dt, tend):
      dnspath = self.respath+"DNS/"
      Re = Constant(self.Re)
      Pr = Constant(self.Pr)
      drup = TrialFunction(self.X)
      s,v,q = TestFunctions(self.X)
      rup = Function(self.X)
      ruppast = Function(self.X)
      ruppastpast = Function(self.X)

      # initialize
      t = 30.
      it = 0
      ifile = 30
 #     U_init = Expression(("0.","0.","0.05*exp(-(x[0]-1)*(x[0]-1)/0.2 - x[1]*x[1]/0.2)","0"), degree=2)
      U_init = Expression(("0.","0.","0.","0."), degree=2)
      File(dnspath+"solution.xml") >> rup.vector()
      ruppast = interpolate(U_init, X)
      ruppast.vector()[:] = rup.vector() + ruppast.vector()
      rho= rup[0]
      u = as_vector((rup[1], rup[2]))
      p = rup[3]
      rhopast= ruppast[0]
      upast = as_vector((ruppast[1], ruppast[2]))
      ppast = ruppast[3]
      
      F = 1./dt*rho*inner(u - upast,v)*dx + 1./dt*(rho - rhopast)*s*dx \
            + 0.5*( \
            + rho*inner( grad(u)*u, v )*dx\
            - div(v)*p*dx + 1./Re*inner(grad(u), grad(v))*dx \
            + dot(grad(rho),u)*s*dx \
            + 1/(Re*Pr)*rho**(-2)*dot(grad(rho), grad(rho*s))*dx
            + rhopast*inner( grad(upast)*upast, v )*dx \
            - div(v)*ppast*dx + 1./Re*inner(grad(upast), grad(v))*dx \
            + dot(grad(rhopast),upast)*s*dx \
            + 1/(Re*Pr)*rhopast**(-2)*dot(grad(rhopast), grad(rhopast*s))*dx \
            ) \
            + rho*div(u)*q*dx - 1./(Re*Pr)*rho**(-2)*dot(grad(rho), grad(rho*q))*dx
      
      rho,u,p = rup.split()
      rho.rename("rho","density"); u.rename("v","velocity");
 #     File(dnspath+"u000.pvd") << u
#      File(dnspath+"rho000.pvd") << rho

      while (t<tend + DOLFIN_EPS):
          print("time t = "+str(np.round(t+dt, decimals=4)))
          dF  = derivative(F, rup, drup) 
          problem = NonlinearVariationalProblem(F, rup, self.bcs, dF)
          # extrapolate for new guess
          rup.vector()[:] = 2*rup.vector() - ruppastpast.vector()
          # solve the problem with Newton
          solver  = NonlinearVariationalSolver(problem)
          solver.parameters['newton_solver']['relative_tolerance']  = 1e-4
          solver.parameters['newton_solver']['absolute_tolerance']  = 1e-4
          solver.parameters['newton_solver']['maximum_iterations']  = 10
          solver.solve()
          ruppastpast.vector()[:]=ruppast.vector()
          ruppast.vector()[:]=rup.vector()
          t = t+dt
          it = it+1
          if (it%10==0):
              ifile=ifile+1
              rho,u,p = rup.split()
              rho.rename("rho","density"); u.rename("v","velocity");
              File(dnspath+"u"+f"{ifile:03d}"+".pvd") << u
              File(dnspath+"rho"+f"{ifile:03d}"+".pvd") << rho
              time.sleep(5)
          
      print("Saving xml file")
      File(dnspath+"solution.xml") << rup.vector()
      rho,u,p = rup.split()
      print("Saving vtk files")
      File(dnspath+"solution_u.pvd") << u
      File(dnspath+"solution_rho.pvd") << rho
      plot(rho)
      
#----------------------------------------------------------------------------------------

   def lowmachTimestepper_CN_Newton_simplifiedDensity(self, dt, tend):
      dnspath = self.respath+"DNS/"
      Re = Constant(self.Re)
      Pr = Constant(self.Pr)
      drup = TrialFunction(self.X)
      s,v,q = TestFunctions(self.X)
      rup = Function(self.X)
      ruppast = Function(self.X)
      ruppastpast = Function(self.X)

      # initialize
      t = 10.
      it = 0
      ifile = 10
#      U_init = Expression(("0.","0.","0.2*exp(-(x[0]-1)*(x[0]-1)/0.2 - x[1]*x[1]/0.2)","0"), degree=2)
      U_init = Expression(("0.","0.","0.","0."), degree=2)
      File(dnspath+"solution.xml") >> rup.vector()
      ruppast = interpolate(U_init, X)
      ruppast.vector()[:] = rup.vector() + ruppast.vector()
      rho= rup[0]
      u = as_vector((rup[1], rup[2]))
      p = rup[3]
      rhopast= ruppast[0]
      upast = as_vector((ruppast[1], ruppast[2]))
      ppast = ruppast[3]
      
      F = 1./dt*rho*inner(u - upast,v)*dx + 1./dt*(rho - rhopast)*s*dx \
            + 0.5*( \
            + rho*inner( grad(u)*u, v )*dx\
            - div(v)*p*dx + 1./Re*inner(grad(u), grad(v))*dx \
            + dot(grad(rho),u)*s*dx + 1/(Re*Pr)*dot(grad(rho), grad(s))*dx \
            + rhopast*inner( grad(upast)*upast, v )*dx \
            - div(v)*ppast*dx + 1./Re*inner(grad(upast), grad(v))*dx \
            + dot(grad(rhopast),upast)*s*dx + 1/(Re*Pr)*dot(grad(rhopast), grad(s))*dx \
            ) \
            + div(u)*q*dx - 1./(Re*Pr)*dot(grad(rho), grad(q))/rho*dx
      
      rho,u,p = rup.split()
      rho.rename("rho","density"); u.rename("v","velocity");
#      File(dnspath+"u000.pvd") << u
#      File(dnspath+"rho000.pvd") << rho

      while (t<tend + DOLFIN_EPS):
          print("time t = "+str(np.round(t+dt, decimals=4)))
          dF  = derivative(F, rup, drup) 
          problem = NonlinearVariationalProblem(F, rup, self.bcs, dF)
          # extrapolate for new guess
          rup.vector()[:] = 2*rup.vector() - ruppastpast.vector()
          # solve the problem with Newton
          solver  = NonlinearVariationalSolver(problem)
          solver.parameters['newton_solver']['relative_tolerance']  = 1e-4
          solver.parameters['newton_solver']['absolute_tolerance']  = 1e-4
          solver.parameters['newton_solver']['maximum_iterations']  = 10
          solver.solve()
          ruppastpast.vector()[:]=ruppast.vector()
          ruppast.vector()[:]=rup.vector()
          t = t+dt
          it = it+1
          if (it%20==0):
              ifile=ifile+1
              rho,u,p = rup.split()
              rho.rename("rho","density"); u.rename("v","velocity");
              File(dnspath+"u"+f"{ifile:03d}"+".pvd") << u
              File(dnspath+"rho"+f"{ifile:03d}"+".pvd") << rho
              time.sleep(5)
          
      print("Saving xml file")
      File(dnspath+"solution.xml") << rup.vector()
      rho,u,p = rup.split()
      print("Saving vtk files")
      File(dnspath+"solution_u.pvd") << u
      File(dnspath+"solution_rho.pvd") << rho
      plot(rho)
      
           
#----------------------------------------------------------------------------------------

   def lowmachTimestepper_EulerNewton(self, dt, tend):
      dnspath = self.respath+"DNS/"
      Re = Constant(self.Re)
      Pr = Constant(self.Pr)
      drup = TrialFunction(self.X)
      s,v,q = TestFunctions(self.X)
      rup = Function(self.X)
      ruppast = Function(self.X)

      # initialize
      t = 0.
      it = 0
      ifile = 0
      U_init = Expression(("1.","1.","0.1*exp(-(x[0]-0.5)*(x[0]-0.5)/0.2 - x[1]*x[1]/0.2)","0"), degree=2)
      rup = interpolate(U_init, X)
      ruppast = interpolate(U_init, X)
#      File(dnspath+"solution.xml") >> rup.vector()
      rho= rup[0]
      u = as_vector((rup[1], rup[2]))
      p = rup[3]
      rhopast= ruppast[0]
      upast = as_vector((ruppast[1], ruppast[2]))
      
      F = 1./dt*rho*inner(u - upast,v)*dx \
            + rho*inner( grad(u)*u, v )*dx\
            - div(v)*p*dx \
            + 1./Re*inner(grad(u), grad(v))*dx \
            + div(u)*q*dx \
            - 1./(Re*Pr)*dot(grad(rho), grad(q))/rho*dx \
            + 1./dt*(rho - rhopast)*s*dx \
            + dot(grad(rho),u)*s*dx \
            + 1/(Re*Pr)*dot(grad(rho), grad(s))*dx
      

      while (t<tend + DOLFIN_EPS):
          print("time t = "+str(np.round(t+dt, decimals=4)))
          dF  = derivative(F, rup, drup) 
          problem = NonlinearVariationalProblem(F, rup, self.bcs, dF)
          # solve the problem with Newton
          solver  = NonlinearVariationalSolver(problem)
          solver.parameters['newton_solver']['relative_tolerance']  = 1e-4
          solver.parameters['newton_solver']['absolute_tolerance']  = 1e-4
          solver.parameters['newton_solver']['maximum_iterations']  = 10
          solver.solve()
          ruppast.vector()[:]=rup.vector()
          t = t+dt
          it = it+1
          if (it%20==0):
              ifile=ifile+1
              rho,u,p = rup.split()
              File(dnspath+"u"+f"{ifile:03d}"+".pvd") << u
              time.sleep(5)
          
      print("Saving xml file")
      File(dnspath+"solution.xml") << rup.vector()
      rho,u,p = rup.split()
      print("Saving vtk files")
      File(dnspath+"solution_u.pvd") << u
      File(dnspath+"solution_rho.pvd") << rho
      plot(rho)
      
           


#----------------------------------------------------------------------------------------

   def lowmachTimestepper_Euler(self, dt, tend):
      dnspath = self.respath+"DNS/"
      Re = Constant(self.Re)
      Pr = Constant(self.Pr)
      rho,u,p = TrialFunctions(self.X)
      s,v,q = TestFunctions(self.X)
      rup = Function(self.X)

      # initialize
      t = 0.
      it = 0
      ifile = 0
      U_init = Expression(("1.","1.","0.05*exp(-(x[0]-1)*(x[0]-1)/0.2 - x[1]*x[1]/0.2)","0"), degree=2)
      rup = interpolate(U_init, X)
#      File(dnspath+"solution.xml") >> rup.vector()
      rhopast= rup[0]
      upast = as_vector((rup[1], rup[2]))
      
 #           - 1./(Re*Pr)*dot(grad(rhopast), grad(q))/rhopast*dx \
      F = 1./dt*rhopast*inner(u - upast,v)*dx \
            + rhopast*inner( grad(upast)*upast, v )*dx\
            - div(v)*p*dx \
            + 1./Re*inner(grad(u), grad(v))*dx \
            + div(u)*q*dx \
            + 1./dt*(rho - rhopast)*s*dx \
            + dot(grad(rhopast),upast)*s*dx \
            + 1/(Re*Pr)*dot(grad(rho), grad(s))*dx
      Aform = lhs(F)
      bform = rhs(F)
      A = assemble(Aform)
      [bc.apply(A) for bc in self.bcs]
#      solver = LUSolver(A)

      while (t<tend + DOLFIN_EPS):
          print("time t = "+str(np.round(t+dt, decimals=4)))
          A = assemble(Aform)
          [bc.apply(A) for bc in self.bcs]
          b = assemble(bform)
          [bc.apply(b) for bc in self.bcs]
          solver = LUSolver(A)
          solver.solve(rup.vector(), b)
          t = t+dt
          it = it+1
          if (it%50==0):
              ifile=ifile+1
              rho,u,p = rup.split()
              File(dnspath+"u"+f"{ifile:03d}"+".pvd") << u
          
      print("Saving xml file")
      File(dnspath+"solution.xml") << rup.vector()
      rho,u,p = rup.split()
      print("Saving vtk files")
      File(dnspath+"solution_u.pvd") << u
      File(dnspath+"solution_rho.pvd") << rho
      plot(rho)
      
           


#----------------------------------------------------------------------------------------

   def nonlinearTimestepper_Euler(self, dt, tend):
      dnspath = self.respath+"DNS/"
      Re = Constant(self.Re)
      u,p = TrialFunctions(self.X)
      v,q = TestFunctions(self.X)
      up = Function(self.X)

      # initialize
      t = 0.
      it = 0
      ifile = 0
      U_init = Expression(("1.","0.05*exp(-(x[0]-0.5)*(x[0]-0.5)/0.2 - x[1]*x[1]/0.2)","0"), degree=2)
      up = interpolate(U_init, X)
#      File("solution.xml") >> up.vector()
      upast = as_vector((up[0], up[1]))
      
      F = 1./dt*inner(u - upast,v)*dx \
            + inner( grad(upast)*upast, v )*dx\
            - div(v)*p*dx \
            + 1./Re*inner(grad(u), grad(v))*dx \
            + div(u)*q*dx
      Aform = lhs(F)
      bform = rhs(F)
      A = assemble(Aform)
      [bc.apply(A) for bc in self.bcs]
      solver = LUSolver(A)

      while (t<tend + DOLFIN_EPS):
          print("time t = "+str(np.round(t+dt, decimals=4)))
          b = assemble(bform)
          [bc.apply(b) for bc in self.bcs]
          solver.solve(up.vector(), b)
          t = t+dt
          it = it+1
          if (it%50==0):
              ifile=ifile+1
              u,p = up.split()
              File(dnspath+"u"+f"{ifile:03d}"+".pvd") << u
          
      print("Saving xml file")
      File(dnspath+"solution.xml") << up.vector()
      u,p = up.split()
      print("Saving vtk files")
      File(dnspath+"solution_u.pvd") << u
      plot(u.sub(0))
      
           
#----------------------------------------------------------------------------------------

   def nonlinearTimestepper_CN(self, dt, tend):
      Re = Constant(self.Re)
      u,p = TrialFunctions(self.X)
      v,q = TestFunctions(self.X)
      up = Function(self.X)
      up_pred  = Function(self.X)
      up_past2 = Function(self.X)

      # initialize
      t = 0.
      it = 0
      ifile = 0
#      U_init = Expression(("1","0","0"), degree=2)
      U_init = Expression(("1.","0.05*exp(-(x[0]-0.5)*(x[0]-0.5)/0.2 - x[1]*x[1]/0.2)","0"), degree=2)
      up = interpolate(U_init, X)
      
      u_past  = as_vector((up[0], up[1]))
      u_pred  = as_vector((up_pred[0], up_pred[1]))
      
      NSOp = 1./dt*inner(u - u_past,v)*dx \
            + inner( grad(u_pred)*u_pred, v )*dx\
            + 0.5/Re*inner(grad(u), grad(v))*dx \
            + 0.5/Re*inner(grad(u_past), grad(v))*dx \
            - div(v)*p*dx \
            + div(u)*q*dx
      Aform = lhs(NSOp)
      bform = rhs(NSOp)
      A = assemble(Aform)
      [bc.apply(A) for bc in self.bcs]
      CNsolver = LUSolver(A)

      while (t<tend + DOLFIN_EPS):
          print("time t = "+str(np.round(t+dt, decimals=4)))
          if (t<dt):
             up_pred.vector()[:] = up.vector()
#             up_pred = up
          else:
             up_pred.vector()[:] = 1.5*up.vector() - 0.5*up_past2.vector()
#             up_pred = 1.5*up - 0.5*up_past2
          up_past2.vector()[:] = up.vector()
#          up_past2 = up
          b = assemble(bform)
          [bc.apply(b) for bc in self.bcs]
          CNsolver.solve(up.vector(), b)
#          u,p = up.split(True)
          t = t+dt
          it = it+1
          if (it%50==0):
              ifile=ifile+1
              u,p = up.split()
              File("u"+f"{ifile:03d}"+".pvd") << u
          
      print("Saving xml file")
      File("solution.xml") << up.vector()
      u,p = up.split()
      print("Saving vtk files")
      File("solution_u.pvd") << u
#      File("solution_p.pvd") << p
      plot(u.sub(0))
      
           
