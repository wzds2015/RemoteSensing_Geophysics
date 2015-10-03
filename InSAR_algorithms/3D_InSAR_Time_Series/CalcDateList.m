function [dayslist,datelist]=CalcDateList(igram)
%
% Extract Datelist from a set of interferograms
%

dim_data=length(igram) ;

for i=1:dim_data
    t(i,:)=[igram(i).t1 igram(i).t2];
end
firsttime=min(min(t));
for i=1:dim_data
   igram(i).t1=igram(i).t1 - firsttime;
   igram(i).t2=igram(i).t2 - firsttime;
    t(i,:)=[igram(i).t1 igram(i).t2];
    %d(i,:)=[igram(i).date1 igram(i).date2];
end
[dayslist,ind]=unique(t);

[Y,X]=L2YX(ind,length(igram));

for ni=1:length(Y) 
    if X(ni)==1  datelist(ni,:)=igram(Y(ni)).date1;
    else         datelist(ni,:)=igram(Y(ni)).date2;
    end
end

