SERVER = "dbmedisightprd\\medisight"
DATABASE = "medisight"
DRIVER = "ODBC Driver 18 for SQL Server"
PORT = "14331"
LARGE_QUERY_CHUNKSIZE = 64

TIME_FORMAT_IN = '%d-%m-%Y'
TIME_FORMAT_QUERY = '%Y-%m-%d'
TIME_FORMAT_QUERY_FULL = '%Y-%m-%dT%H:%M:%S'


Q_CONSULT_DATES_AND_DOCTORS = '''select distinct pat_id,ety_fullname,pat_surname,pat_forename,enc.ect_date, u.use_fullname DocumentLastSavedBy, pt.acp_desc, uep.use_fullname
from (
    select distinct ect_id, enc.ect_pat_id, enc.ect_brf_BillingReferenceId, enc.ect_date, ety.ety_fullname, ect_daterecorded, ect_datesaved, ect_use_id_lastsaved
    from dbo.Encounter enc
    inner JOIN dbo.encountertype ety ON enc.ect_ety_id = ety.ety_id
    where
   	 ety.ety_fullname IN ('Consultation', 'Opération')
   	 AND enc.ect_isdeleted = 0 --Rencontre valide
     AND enc.ect_date between '{start}' and '{end}'
   	 --and ect_pat_id = 'PUT_SOMETHING_HERE_IF_NEED_FILT'
) enc
inner join dbo.Patient ms_p on enc.ect_pat_id = ms_p.pat_id
left join Billing.reference pr on pr.brf_id = enc.ect_brf_BillingReferenceId
left join ClinicList.PatientIdentification pi on ms_p.pat_id = pi.clp_pat_id
/* get doctor information */
left join dbo."User" u ON enc.ect_use_id_lastsaved = u.use_id
left join dbo.encounterpersonnel ep ON enc.ect_id = ep.enp_ect_id
    left join dbo."User" uep ON ep.enp_use_id = uep.use_id
   	 left join dbo.personneltype pt on ep.enp_acp_id = pt.acp_id --Fonction ou rôle, acp_desc-> "Vu par": person to which we want to award stars; "Clinicien responsable": supervisor
where pt.acp_desc = 'Vu par' or pt.acp_desc = 'Chirurgien'
'''

Q_DIAGNOSES_XML = '''SELECT pto_overview,* FROM [dbo].patientoverview o
inner join dbo.patient p on o.pto_pat_id = p.pat_id
left join mediSIGHT_reporting.dbo.Patient_Details p2 on o.pto_pat_id = p2.PatientId
where 1=1
and pto_pat_id = '%s'
and pto_widget_id=308 --CurrentProblemWidget=Diagnostics --En cours
'''

Q_DOCS_AND_POSITIONS = '''select distinct u.use_id, u.use_username, u.use_domainuser, u.use_givenname, u.use_familyname, u.use_fullname
       , uas.uas_description
       , gra.gra_desc, gra.gra_desclong
       , ust.ust_desc, ust.ust_valid
       , grp.grp_name, grp_active, grp_description
from dbo."User" u 
inner join dbo.UserAccountStatus uas on u.use_uas_id = uas.uas_id
left join dbo.grade gra on u.use_gra_id = gra.gra_id
inner join dbo.usertype ust on u.use_ust_id = ust.ust_id
inner join dbo.groupstousers ug on u.use_id = ug.user_id
inner join dbo.groups grp on grp.grp_id = ug.groups_id
where gra.gra_desc like '%assistant%' and gra.gra_desc not like '%chef%'
order by use_fullname
'''

Q_DOCS_AND_POSITIONS_FULL = '''select distinct u.use_id, u.use_username, u.use_domainuser, u.use_givenname, u.use_familyname, u.use_fullname
       , uas.uas_description
       , gra.gra_desc, gra.gra_desclong
       , ust.ust_desc, ust.ust_valid
       , grp.grp_name, grp_active, grp_description
from dbo."User" u 
inner join dbo.UserAccountStatus uas on u.use_uas_id = uas.uas_id
left join dbo.grade gra on u.use_gra_id = gra.gra_id
inner join dbo.usertype ust on u.use_ust_id = ust.ust_id
inner join dbo.groupstousers ug on u.use_id = ug.user_id
inner join dbo.groups grp on grp.grp_id = ug.groups_id
order by use_fullname
'''