function M = Make_designMatrix_3D(igram,before1,after1)
%
% Generate Design Matrix 
%
% [A,B,C1,C2,C3,covD,covV]=MakeDesignMatrix_v1(igram);
%
% igram1 = igram(1,1:N_igrams1);
% [dayslist1 datelist1]=CalcDateList(igram1);
N_igrams=length(igram) ;
% for ni=N_igrams1+1:N_igrams
% %     igram(ni).t1 = igram(ni).t1 + dayslist1(5) + 1;
% %     igram(ni).t2 = igram(ni).t2 + dayslist1(5) + 1;
%     igram(ni).t1 = igram(ni).t1 + dayslist1(5);
%     igram(ni).t2 = igram(ni).t2 + dayslist1(5);
% end
% clear igram1
[dayslist datelist] = CalcDateList(igram);
N_timevector=length(dayslist);

A=single(zeros([N_igrams,N_timevector*2-before1-after1]));
B=single(zeros(size(A)));
for ni=1:N_igrams
   ind_t1=find(dayslist==igram(ni).t1);
   ind_t2=find(dayslist==igram(ni).t2);
   if  ind_t1 <= before1
       if ind_t2 <=before1
           B(ni,ind_t1:ind_t2-1)=1;
       elseif ind_t2 <= length(datelist) - after1+1 
           B(ni,ind_t1:(ind_t2 - 1)*2 - before1)=1;
       else
           B(ni,ind_t1:ind_t2*2-before1-after1*2 + (after1-(length(datelist)-ind_t2)))=1;
       end
   elseif ind_t1 <= length(datelist) - after1
       if ind_t2 <= length(datelist) - after1+1
           B(ni,(ind_t1-1)*2-before1+1:(ind_t2-1)*2-before1)=1;
       else
           B(ni,(ind_t1-1)*2-before1+1:ind_t2*2-before1-after1*2 + (after1-(length(datelist)-ind_t2)))=1;
       end
   else
       B(ni,ind_t1*2-before1-after1*2 + (after1-(length(datelist)-ind_t1)):ind_t2*2-before1-after1*2 + (after1-(length(datelist)-ind_t2)))=1;
           %  for range change velocity
   end
end


% A(:,1)=[];               % this corresponds to removing the column of A belonging to t_0
B(:,end)=[];
temp1 = single(ones(1,N_timevector*2-before1-after1-1));
for ni=1:before1
    temp1(ni) = dayslist(ni+1)-dayslist(ni);
end
for ni=(before1+1):2:(N_timevector*2-before1-after1*2)
    temp1(ni) = dayslist((ni+1-before1)/2+before1+1) - dayslist((ni+1-before1)/2+before1);
    temp1(ni+1) = temp1(ni);
end
for ni = N_timevector*2-before1-after1*2+1:N_timevector*2-before1-after1-1
    temp1(ni) = dayslist(N_timevector-(N_timevector*2-before1-after1-1-ni)) - dayslist(N_timevector-(N_timevector*2-before1-after1-1-ni)-1);
end
M = single(zeros(N_igrams,N_timevector*2-before1-after1-1));
for ni = 1:N_igrams
    M(ni,:) = B(ni,:).*temp1;
end
