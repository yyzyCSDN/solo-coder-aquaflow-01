from .treatment.chlorine_ct import compliant
from .network.isolation import isolated
from .meters.balance import balance
from .leak.localize import rank
from .workflow_service import WorkflowService
from .treatment.chlorine_desk import ChlorineDesk, Policy as ChlorinePolicy
class AquaFlowService:
    def __init__(self):self.workflows=WorkflowService();self._desks={}
    def health(self):return {"service":"aquaflow","status":"ok","runtime_dependencies":0}
    def ct(self,rows,required):return compliant(rows,required)
    def isolate(self,nodes,pipes,closed,sources):return isolated(nodes,pipes,closed,sources)
    def water_balance(self,ins,outs,known_loss):return balance(ins,outs,known_loss)
    def leak_rank(self,rows):return rank(rows)
    def run_workflow(self,name,payload,received_at=None):return self.workflows.run(name,payload,received_at)
    def chlorine_desk(self,desk_id="default",**policy):
        desk=self._desks.get(desk_id)
        if desk is None:
            desk=ChlorineDesk(ChlorinePolicy(**policy) if policy else None);self._desks[desk_id]=desk
        return desk
    def ingest_chlorine_readings(self,rows,desk_id="default",received_at=None,chain=None):return self.chlorine_desk(desk_id).ingest(rows,received_at,chain)
    def evaluate_chlorine_window(self,chain,start,end,desk_id="default",at=None):return self.chlorine_desk(desk_id).evaluate(chain,start,end,at)
    def decide_chlorine_case(self,case_id,action,reason,actor="operator",at=None,desk_id="default"):return self.chlorine_desk(desk_id).decide(case_id,action,reason,actor,at)
    def chlorine_case(self,case_id,desk_id="default"):return self.chlorine_desk(desk_id).case_view(case_id)
    def chlorine_cases(self,desk_id="default"):return self.chlorine_desk(desk_id).cases()
    def replay_chlorine_case(self,case_id,desk_id="default"):return self.chlorine_desk(desk_id).replay(case_id)
