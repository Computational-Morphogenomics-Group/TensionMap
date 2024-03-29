function [theta, ERes] = theta_optimization(x0, q, p, d0, bCells, rBX, rBY)
    p = p';
    x0 = x0';

    energyFunc = @(x) thetaEnergy(x, q, p, d0, bCells, rBX, rBY);
    if (energyFunc(zeros(size(x0))) < energyFunc(x0))
        x0 = zeros(size(x0));
    end

    Aeq = ones(1,length(x0));
    beq = 0;
    
    dP = p(bCells(:,1)) - p(bCells(:,2));
    A = bsxfun(@times,dP,d0);
    dQ = q(bCells(:,1),:) - q(bCells(:,2),:);
    QL = sum(dQ.^2,2);
    b = p(bCells(:,1)).*p(bCells(:,2)).*QL;

    optimset = optimoptions('fmincon','Display','none','Algorithm','interior-point','TolFun',1e-6, ...
                            'MaxFunEvals',5e6,'MaxIter',2e3,'GradObj','on','GradConstr','on',...,
                            'HessianApproximation','lbfgs');

    [theta,ERes] = fmincon(energyFunc,x0,A,b,Aeq,beq,[],[],[],optimset);
end

function [ E, dE ] = thetaEnergy( theta, q, p, d0, bCells, RBx, RBy )
    % REDUCED ENERGY 

    dP = p(bCells(:,1))-p(bCells(:,2));
    dT = theta(bCells(:,1))-theta(bCells(:,2));
    dQ = q(bCells(:,1),:)-q(bCells(:,2),:);
    QL = sum(dQ.^2,2);
    
    rho = bsxfun(@rdivide,d0*bsxfun(@times,p,q),dP);
    Rsq = ( p(bCells(:,1)).*p(bCells(:,2)).*QL - (dP .* dT) ) ./ dP.^2;

    R = sqrt(Rsq);
    indZ = Rsq<=0;
    R(indZ) = 0;
    
    deltaX = -bsxfun(@minus,RBx,rho(:,1));
    deltaY = -bsxfun(@minus,RBy,rho(:,2));
    dMag = sqrt(deltaX.^2 + deltaY.^2);
    d = bsxfun(@minus,dMag,R);

    E = .5*mean( sum(d.^2,2) );
    
    if (nargout == 2)
        
        [ dR ] = returnBondGradsTheta( q, theta, p, bCells );

        avgD = sum(d,2);
        NB = size(d0,1);
        NC = size(d0,2);
        dR(indZ,:) = 0;

        dE = -bsxfun(@times,avgD,dR) / NB;
        rows = [bCells(:,1);bCells(:,2)];
        dE = accumarray(rows,dE(:),[NC,1]);
    end
    
end

function [ dR ] = returnBondGradsTheta( q, theta, p, bCells )
    % RETURN BOND GRADS 

    p1 = p(bCells(:,1));
    p2 = p(bCells(:,2));
    q1x = q(bCells(:,1),1);
    q2x = q(bCells(:,2),1);
    q1y = q(bCells(:,1),2);
    q2y = q(bCells(:,2),2);
    t1 = theta(bCells(:,1));
    t2 = theta(bCells(:,2));
    
    dR = radiusGradTheta(p1,p2,q1x,q2x,q1y,q2y,t1,t2);
end

function Dr = radiusGradTheta(p1,p2,q1x,q2x,q1y,q2y,t1,t2)
    %RADIUSGRADTHETA
    %    DR = RADIUSGRADTHETA(P1,P2,Q1X,Q2X,Q1Y,Q2Y,T1,T2)
    
    %    This function was generated by the Symbolic Math Toolbox version 5.10.
    %    06-Jul-2017 12:09:33
    
    t4 = p1-p2;
    t5 = q1x-q2x;
    t6 = q1y-q2y;
    t7 = 1.0./t4.^2;
    t8 = t1-t2;
    t9 = t4.*t8;
    t10 = t5.^2;
    t11 = t6.^2;
    t12 = t10+t11;
    t13 = t9-p1.*p2.*t12;
    t14 = 1.0./sqrt(-t7.*t13);
    t15 = 1.0./t4;
    Dr = [t14.*t15.*(-1.0./2.0),t14.*t15.*(1.0./2.0)];
end

