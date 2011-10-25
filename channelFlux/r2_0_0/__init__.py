#!/usr/bin/env python

#--------------------------------------------------------------------------------------
## pythonFlu - Python wrapping for OpenFOAM C++ API
## Copyright (C) 2010- Alexey Petrov
## Copyright (C) 2009-2010 Pebble Bed Modular Reactor (Pty) Limited (PBMR)
## 
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.
## 
## See http://sourceforge.net/projects/pythonflu
##
## Author : Alexey PETROV
##


#----------------------------------------------------------------------------
from Foam import ref, man

#----------------------------------------------------------------------------
def readTransportProperties( runTime, mesh):
    ref.ext_Info() << "\nReading transportProperties\n" << ref.nl
    
    transportProperties = man.IOdictionary( man.IOobject( ref.word( "transportProperties" ),
                                                          ref.fileName( runTime.constant() ),
                                                          mesh,
                                                          ref.IOobject.MUST_READ_IF_MODIFIED,
                                                          ref.IOobject.NO_WRITE,
                                                          False ) )
    
    nu = ref.dimensionedScalar( transportProperties.lookup( ref.word( "nu" ) ) )
    
    #  Read centerline velocity for channel simulations
    Ubar = ref.dimensionedVector( transportProperties.lookup( ref.word( "Ubar" ) ) )

    magUbar = Ubar.mag()
    flowDirection = ( Ubar / magUbar ).ext_value()
    
    return transportProperties, nu, Ubar, magUbar, flowDirection


#----------------------------------------------------------------------------
def _createFields( runTime, mesh ):
        
    ref.ext_Info() << "Reading field p\n" << ref.nl
    p = man.volScalarField( man.IOobject( ref.word( "p" ),
                                          ref.fileName( runTime.timeName() ),
                                          mesh,
                                          ref.IOobject.MUST_READ,
                                          ref.IOobject.AUTO_WRITE ),
                            mesh )

    ref.ext_Info() << "Reading field U\n" << ref.nl

    U = man.volVectorField( man.IOobject( ref.word( "U" ),
                                          ref.fileName( runTime.timeName() ),
                                          mesh,
                                          ref.IOobject.MUST_READ,
                                          ref.IOobject.AUTO_WRITE ),
                            mesh )
    
    phi = man.createPhi( runTime, mesh, U )
    
    pRefCell = 0
    pRefValue = 0.0
    
    pRefCell, pRefValue = ref.setRefCell( p, mesh.solutionDict().subDict( ref.word( "PISO" ) ), pRefCell, pRefValue )

    laminarTransport = man.singlePhaseTransportModel( U, phi )
    
    sgsModel = man.incompressible.LESModel.New( U, phi, laminarTransport )
        
    return p, U, phi, laminarTransport, sgsModel, pRefCell, pRefValue


#--------------------------------------------------------------------------------------
def createGradP( runTime ):
    gradP = ref.dimensionedScalar( ref.word( "gradP" ),
                                   ref.dimensionSet( 0.0, 1.0, -2.0, 0.0, 0.0 ),
                                   0.0 )

    gradPFile = ref.IFstream( runTime.path()/ref.fileName( runTime.timeName() )/ref.fileName( "uniform" )/ ref.fileName( "gradP.raw" ) )
    
    if gradPFile.good():
       gradPFile >> gradP
       ref.ext_Info() << "Reading average pressure gradient" << ref.nl << ref.nl
       pass
    else:
       ref.ext_Info() << "Initializing with 0 pressure gradient" << ref.nl << ref.nl
       pass
    
    return gradP, gradPFile


#-------------------------------------------------------------------------------------
def writeGradP( runTime, gradP ):
    if runTime.outputTime():
       gradPFile = ref.OFstream( runTime.path()/ref.fileName( runTime.timeName() )/ref.fileName( "uniform" )/ref.fileName( "gradP.raw" ) )
       
       if gradPFile.good():
          gradPFile << gradP << ref.nl
          pass
       else:
          ref.ext_Info() << "Cannot open file " << runTime.path()/ref.fileName( runTime.timeName() )/ref.fileName( "uniform" )/ref.fileName( "gradP.raw" )
          import os
          os.abort()

#--------------------------------------------------------------------------------------
def main_standalone( argc, argv ):

    args = ref.setRootCase( argc, argv )

    runTime = man.createTime( args )

    mesh = man.createMesh( runTime )
    
    transportProperties, nu, Ubar, magUbar, flowDirection = readTransportProperties( runTime, mesh)
    
    p, U, phi, laminarTransport, sgsModel, pRefCell, pRefValue = _createFields( runTime, mesh )
    
    cumulativeContErr = ref.initContinuityErrs()
    
    gradP, gradPFile = createGradP( runTime)
    
    ref.ext_Info() << "\nStarting time loop\n" << ref.nl 

    while runTime.loop() :
        ref.ext_Info() << "Time = " << runTime.timeName() << ref.nl << ref.nl

        piso, nCorr, nNonOrthCorr, momentumPredictor, transonic, nOuterCorr = ref.readPISOControls( mesh ) 

        CoNum, meanCoNum = ref.CourantNo( mesh, phi, runTime )

        sgsModel.correct()

        UEqn = ref.fvm.ddt( U ) + ref.fvm.div( phi, U ) + sgsModel.divDevBeff( U ) == flowDirection * gradP

        if momentumPredictor:
           ref.solve( UEqn == -ref.fvc.grad( p ) )
           pass

        rAU = 1.0 / UEqn.A()

        for corr in range( nCorr ):
            U << rAU * UEqn.H()

            phi << ( ref.fvc.interpolate( U ) & mesh.Sf() ) + ref.fvc.ddtPhiCorr( rAU, U, phi )

            ref.adjustPhi(phi, U, p)

            for nonOrth in range( nNonOrthCorr + 1 ):
                pEqn = ref.fvm.laplacian( rAU, p ) == ref.fvc.div( phi ) 
                pEqn.setReference( pRefCell, pRefValue )

                if corr == nCorr-1 and nonOrth == nNonOrthCorr:
                   pEqn.solve( mesh.solver( ref.word( str( p.name() ) + "Final" ) ) )
                   pass
                else:
                   pEqn.solve( mesh.solver( p.name() ) )
                   pass

                if nonOrth == nNonOrthCorr:
                   phi -= pEqn.flux()
                   pass
                pass

            cumulativeContErr = ref.ContinuityErrs( phi, runTime, mesh, cumulativeContErr )

            U -= rAU * ref.fvc.grad( p )
            U.correctBoundaryConditions()
            pass

        # Correct driving force for a constant mass flow rate

        # Extract the velocity in the flow direction
        magUbarStar = ( flowDirection & U ).weightedAverage( mesh.V() )

        # Calculate the pressure gradient increment needed to
        # adjust the average flow-rate to the correct value
        gragPplus = ( magUbar - magUbarStar ) / rAU.weightedAverage( mesh.V() )

        U << U() + flowDirection * rAU * gragPplus # mixed caculations

        gradP +=gragPplus
        ref.ext_Info() << "Uncorrected Ubar = " << magUbarStar.value() << " " << "pressure gradient = " << gradP.value() << ref.nl

        runTime.write()

        writeGradP( runTime, gradP )

        ref.ext_Info() << "ExecutionTime = " << runTime.elapsedCpuTime() << " s" << \
              "  ClockTime = " << runTime.elapsedClockTime() << " s" << ref.nl << ref.nl

        pass

    ref.ext_Info() << "End\n" << ref.nl 

    import os
    return os.EX_OK


#--------------------------------------------------------------------------------------
from Foam import FOAM_VERSION
if FOAM_VERSION( ">=", "020000" ):
   if __name__ == "__main__" :
     import sys, os
     argv = sys.argv
     os._exit( main_standalone( len( argv ), argv ) )
     pass
else:
   ref.ext_Info() << "\n\n To use this solver it is necessary to SWIG OpenFOAM-2.0.0 or higher\n"
   pass 
   
     
#--------------------------------------------------------------------------------------

