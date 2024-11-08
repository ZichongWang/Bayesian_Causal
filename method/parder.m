function [grad] = parder(y, qBD, qLS, qLF, alLS, alLF, w, local)

    %%  Guide 
    %   w has [w0;weps;w0BD;w0LS;w0LF;wLSBD;wLFBD;wBDy;wLSy;wLFy;weLS;weLF;weBD;waLS;waLF] 
    %   grad is a matrix of row locations for the batch size and 15 weights
    y=log(1e-6 + y);
    
    %% w0y
    grad(:,1)   =                                   ( 1./(w(2).^2) ).*( y - w(1) ) ...
                -  ( local == 3 | local == 4 | local == 6)  .*  ( 1./(w(2).^2) ).*( w(8).*qBD ) ...
                -  ( local == 1 | local == 3 | local == 5 | local == 6)  .*  ( 1./(w(2).^2) ).*( w(9).*qLS ) ...
                -  ( local == 2 | local == 4 | local == 5 | local == 6)  .*  ( 1./(w(2).^2) ).*( w(10).*qLF );
    
    %% wey
    grad(:,2)   =   - 1 ./ w(2) - (1./w(2).^3) .* (- y.^2 - w(1).^2 + 2.*w(1).*y ...
                                                   - (local == 1 | local == 3 | local == 5 | local == 6) .* (w(9).^2  - 2 .* y .* w(9)  + 2 .* w(1) .* w(9))  .* qLS ...
                                                   - (local == 2 | local == 4 | local == 5 | local == 6) .* (w(10).^2 - 2 .* y .* w(10) + 2 .* w(1) .* w(10)) .* qLF ...
                                                   - (local == 3 | local == 4 | local == 6)              .* (w(8).^2  - 2 .* y .* w(8)  + 2 .* w(1) .* w(8))  .* qBD ...
                                                   - (local == 3 ) .* 2 .* (w(8) .* w(9) .* qLS .* qBD) ...
                                                   - (local == 4 ) .* 2 .* (w(8) .* w(10) .* qLF .* qBD) ...
                                                   - (local == 6 ) .* 2 .* (w(8) .* w(8) .* w(10) .* qLS .* qLF .* qBD));

    %% w0BD
    grad(:,3)   =  ( local == 3 )  .*  ( qBD     .* qLS     .* df(  w(3) + w(6) - (w(13).^2)./2)...
                                       - (1-qBD) .* qLS     .* df(- w(3) - w(6) - (w(13).^2)./2) ...
                                       + qBD     .* (1-qLS) .* df(  w(3)        - (w(13).^2)./2)...
                                       - (1-qBD) .* (1-qLS) .* df(- w(3)        - (w(13).^2)./2))...
                +  ( local == 4 )  .*  ( qBD     .* qLF     .* df(  w(3) + w(7) - (w(13).^2)./2)...
                                       - (1-qBD) .* qLF     .* df(- w(3) - w(7) - (w(13).^2)./2) ...
                                       + qBD     .* (1-qLF) .* df(  w(3)        - (w(13).^2)./2)...
                                       - (1-qBD) .* (1-qLF) .* df(- w(3)        - (w(13).^2)./2)) ...
                +  ( local == 6 )  .*  ( qBD     .* qLS     .* qLF     .* df(w(3) + w(6) + w(7) - (w(13).^2)./2)...
                                       + qBD     .* qLS     .* (1-qLF) .* df(w(3) + w(6)        - (w(13).^2)./2) ...
                                       + qBD     .* (1-qLS) .* qLF     .* df(w(3) + w(7)        - (w(13).^2)./2)...
                                       + qBD     .* (1-qLS) .* (1-qLF) .* df(w(3)               - (w(13).^2)./2))...
                +  ( local == 6 )  .* (- (1-qBD) .* qLS     .* qLF     .* df(- w(3) - w(6) - w(7) - (w(13).^2)./2)...
                                       - (1-qBD) .* qLS     .* (1-qLF) .* df(- w(3) - w(6)        - (w(13).^2)./2) ...
                                       - (1-qBD) .* (1-qLS) .* qLF     .* df(- w(3) - w(7)        - (w(13).^2)./2)...
                                       - (1-qBD) .* (1-qLS) .* (1-qLF) .* df(- w(3)               - (w(13).^2)./2));
            
    %% w0LS  
    grad(:,4)   =  ( local == 1 | local == 3 | local == 5 | local == 6)  .*  qLS.*df(w(4) + w(14) .* alLS - (w(11).*w(11)/2)) ...
                -  ( local == 1 | local == 3 | local == 5 | local == 6)  .*  (1-qLS).*df(- w(4) - w(14) .* alLS - (w(11).*w(11)/2));
            
    %% w0LF
    %% 检查
    grad(:,5)   =  ( local == 2 | local == 4 | local == 5 | local == 6)  .*  qLF.*df(w(5) + w(15) .* alLF - (w(12).*w(12)/2)) ...
                -  ( local == 2 | local == 4 | local == 5 | local == 6)  .*  (1-qLF).*df(- w(5) - w(15) .* alLF - (w(12).*w(12)/2));
             
    %% wLSBD       
    grad(:,6)   =  ( local == 3 ) .* ( qBD    .* qLS .*  df(  w(3) + w(6) - (w(13).^2)./2) ...
                                     - (1-qBD).* qLS .*  df(- w(3) - w(6) - (w(13).^2)./2))...
                +  ( local == 6 ) .* ( qBD    .* qLS .* qLF     .* df(  w(3) + w(6) + w(7) - (w(13).^2)./2) ...
                                     + qBD    .* qLS .* (1-qLF) .* df(  w(3) + w(6)        - (w(13).^2)./2) ...
                                     - (1-qBD).* qLS .* qLF     .* df(- w(3) - w(6) - w(7) - (w(13).^2)./2)...
                                     - (1-qBD).* qLS .* (1-qLF) .* df(- w(3) - w(6)        - (w(13).^2)./2));
    
    %% wLFBD
    grad(:,7)   =  ( local == 4 ) .* ( qBD    .* qLF .* df(  w(3) + w(7) - (w(13).^2)./2) ...
                                     - (1-qBD).* qLF .* df(- w(3) - w(7) - (w(13).^2)./2))...
                +  ( local == 6 ) .* ( qBD    .* qLF .* qLS     .* df(  w(3) + w(6) + w(7) - (w(13).^2)./2) ...
                                     + qBD    .* qLF .* (1-qLS) .* df(  w(3) + w(7)        - (w(13).^2)./2) ...
                                     - (1-qBD).* qLF .* qLS     .* df(- w(3) - w(6) - w(7) - (w(13).^2)./2)...
                                     - (1-qBD).* qLF .* (1-qLS) .* df(- w(3) - w(7)        - (w(13).^2)./2));
    
	%% wBDy
    grad(:,8)   =  ( local == 3 | local == 4 | local == 6)  .*  (- 1./(w(2).^2)).*qBD.*(w(8) - y + w(1)) ...
                - (1./(w(2).^2)) ...
                .*(( local == 3 ) .* w(9)          .* qBD .* qLS ...
                +  ( local == 4 ) .* w(10)         .* qBD        .* qLF ...
                +  ( local == 6 ) .* w(9) .* w(10) .* qBD .* qLS .* qLF );
    
	%% wLSy
    grad(:,9)   =  ( local == 1 | local == 3 | local == 5 | local == 6)  .*  (- 1./(w(2)^2)).*qLS.*(w(9) - y + w(1)) ...
                - (1./(w(2)^2)) ...
                .*(( local == 3 ) .* w(8)          .* qBD .* qLS ...
                +  ( local == 6 ) .* w(8) .* w(10) .* qBD .* qLS .* qLF);
    
	%% wLFy
    grad(:,10)  =  ( local == 2 | local == 4 | local == 5 | local == 6)  .*  (- 1./(w(2)^2)).*qLF.*(w(10) - y + w(1)) ...
                - (1./(w(2)^2)) ...
                .*(( local == 4 ) .* w(8)          .* qBD .* qLF ...
                +  ( local == 6 ) .* w(8) .* w(9) .* qBD .* qLF .* qLS);
            
    %% weLS
    grad(:,11)  =  ( local == 1 | local == 3 | local == 5 | local == 6)  .*  (- qLS .* w(11) .* df(w(4) + w(14) .* alLS - (w(11).*w(11)./2))...
                                                     - (1 - qLS) .* w(11) .* df(- w(4) - w(14) .* alLS - (w(11).*w(11)./2)));
                                                 
    %% weLF
    grad(:,12)  =  ( local == 2 | local == 4 | local == 5 | local == 6)  .*  (- qLF .* w(12) .* df(w(5) + w(15) .* alLF - (w(12).*w(12)./2))...
                                                     - (1 - qLF) .* w(12) .* df(- w(5) - w(15) .* alLF - (w(12).*w(12)./2)));
                                                 
    %% weBD
    grad(:,13) = ( local == 3 )  .* (- qBD     .* qLS     .* w(13) .* df(  w(3) + w(6) - (w(13).^2)./2)...
                                     - (1-qBD) .* qLS     .* w(13) .* df(- w(3) - w(6) - (w(13).^2)./2) ...
                                     - qBD     .* (1-qLS) .* w(13) .* df  (w(3)        - (w(13).^2)./2)...
                                     - (1-qBD) .* (1-qLS) .* w(13) .* df(- w(3)        - (w(13).^2)./2))...
               + ( local == 4 )  .* (- qBD     .* qLF     .* w(13) .* df(  w(3) + w(7) - (w(13).^2)./2)...
                                     - (1-qBD) .* qLF     .* w(13) .* df(- w(3) - w(7) - (w(13).^2)./2) ...
                                     - qBD     .* (1-qLF) .* w(13) .* df(  w(3)        - (w(13).^2)./2)...
                                     - (1-qBD) .* (1-qLF) .* w(13) .* df(- w(3)        - (w(13).^2)./2)) ...
               + ( local == 6 )  .* (- qBD     .* qLS     .* qLF     .* w(13) .* df(  w(3) + w(6) + w(7) - (w(13).^2)./2)...
                                     - qBD     .* qLS     .* (1-qLF) .* w(13) .* df(  w(3) + w(6)        - (w(13).^2)./2)...
                                     - qBD     .* (1-qLS) .* qLF     .* w(13) .* df(  w(3)        + w(7) - (w(13).^2)./2)...
                                     - qBD     .* (1-qLS) .* (1-qLF) .* w(13) .* df(  w(3)               - (w(13).^2)./2)...
                                     - (1-qBD) .* qLS     .* qLF     .* w(13) .* df(- w(3) - w(6) - w(7) - (w(13).^2)./2) ...
                                     - (1-qBD) .* qLS     .* (1-qLF) .* w(13) .* df(- w(3) - w(6)        - (w(13).^2)./2) ...
                                     - (1-qBD) .* (1-qLS) .* qLF     .* w(13) .* df(- w(3)        - w(7) - (w(13).^2)./2) ...
                                     - (1-qBD) .* (1-qLS) .* (1-qLF) .* w(13) .* df(- w(3)               - (w(13).^2)./2)) ;
                                 
     %% waLS
     grad(:,14) = ( local == 1| local == 3 | local == 5 | local == 6) .* (qLS .* alLS .* df(w(4) + w(14) .* alLS - (w(11).^2)./2) ...
                                                - (1 - qLS) .* alLS .* df(- w(4) - w(14) .* alLS - (w(11).^2)/2) );
                                            
     %% waLF
     grad(:,15) = ( local == 2| local == 4 | local == 5 | local == 6) .* (qLF .* alLF .* df(w(5) + w(15) .* alLF - (w(12).^2)./2) ...
                                                - (1 - qLF) .* alLF .* df(- w(5) - w(15) .* alLF - (w(12).^2)/2) );
                                            
%% test 
if sum(sum(isnan(grad)))>0
    grad;
end
end
