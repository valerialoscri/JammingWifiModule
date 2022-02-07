#include "gym-actor-critic.h"
#include "ns3/object.h"
#include "ns3/core-module.h"
#include "ns3/wifi-module.h"
#include "ns3/node-list.h"
#include "ns3/log.h"
#include <sstream>
#include <iostream>


namespace ns3 {

    NS_LOG_COMPONENT_DEFINE ("GymActorCritic");

    NS_OBJECT_ENSURE_REGISTERED (GymActorCritic);

    TypeId
    GymActorCritic::GetTypeId(void)
    {
        static TypeId tid = TypeId("ns3::GymActorCritic")
        .SetParent<OpenGymEnv> ()
        .SetGroupName("OpenGym")
        .AddConstructor <GymActorCritic>()
        ;
     return tid;
    }

    GymActorCritic::GymActorCritic()
    {
        NS_LOG_FUNCTION(this);
        m_currentNode=0;
        m_currentChannel=0;
        m_collisionTh=225;
        m_channelNum =1;
       // m_channelOccupation.clear();
    }

    GymActorCritic::~GymActorCritic()
    {
        NS_LOG_FUNCTION(this);
    }

    GymActorCritic::GymActorCritic(uint32_t currentChannel,uint32_t channelNum)
    {
        NS_LOG_FUNCTION(this);
        m_currentNode=0;
        m_currentChannel = currentChannel;
        m_collisionTh =22;
        m_channelNum=12;
        //m_channelOccupation.clear();

    }

    void
    GymActorCritic::DoDispose()
    {
        NS_LOG_FUNCTION(this);
    }

    uint32_t 
    GymActorCritic::GetCurrentChannel()
    {
        NS_LOG_FUNCTION(this);
        return m_currentChannel;
    }
    
    void
    GymActorCritic::SetCurrentChannel(uint32_t channel)
    {
     NS_LOG_FUNCTION(this);  
     m_currentChannel = channel; 
    }

    Ptr<OpenGymSpace>
    GymActorCritic::GetObservationSpace()
    {
        NS_LOG_FUNCTION(this);
        float low = 0.0;
        float high = 1.0;
        std::vector<uint32_t> shape = {m_channelNum,};
        std::string dtype = TypeNameGet<uint32_t>();
        Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace>(low,high,shape,dtype);
        NS_LOG_UNCOND ("GetObservationsSpace: " << space);
        return space;
    }


    Ptr<OpenGymSpace>
    GymActorCritic::GetActionSpace()
    {
        NS_LOG_FUNCTION(this);
        
        Ptr<OpenGymDiscreteSpace> space = CreateObject<OpenGymDiscreteSpace>(m_channelNum);
        NS_LOG_UNCOND ("GetActionSpace:" << space);
        return space;
    }


    bool
    GymActorCritic::GetGameOver(){
        NS_LOG_FUNCTION(this);
        bool isGameOver = false;

        uint32_t collisionNum = 0;
        for(auto&v :m_collision){
            collisionNum +=v;
        }

        if(collisionNum >= m_collisionTh){
            isGameOver=true;
        }
        return isGameOver;

    }

    Ptr<OpenGymDataContainer>
    GymActorCritic::GetObservation()
    {
        NS_LOG_FUNCTION(this);
        std::vector<uint32_t> shape = {m_channelNum,};
        Ptr<OpenGymBoxContainer<uint32_t> > box = CreateObject<OpenGymBoxContainer<uint32_t> >(shape);

        for(uint32_t i =0; i<m_channelOccupation.size(); i++){
            uint32_t value = m_channelOccupation.at(i);
            box->AddValue(value);
        }
        NS_LOG_UNCOND("myGetObservation"<<box);
        return box;
    }


    float
    GymActorCritic::GetReward()
    {
        NS_LOG_UNCOND(this);
        float reward = 1.0;
        if(m_channelOccupation.size() == 0){
            return 0.0;
        }

        uint32_t occupied = m_channelOccupation.at(m_currentChannel);
        if(occupied == 2){
            m_collision.erase(m_collision.begin());
            m_collision.push_back(0);
            
        }
        else{
            reward = -1;
            m_collision.erase(m_collision.begin());
            m_collision.push_back(1);
        }

        NS_LOG_UNCOND("Reward"<< reward);
        return reward;
    }


    std::string
    GymActorCritic::GetExtraInfo()
    {
        NS_LOG_FUNCTION(this);
        std::string myInfo = "info";
        return myInfo;
    }

    bool
    GymActorCritic::ExecuteActions(Ptr<OpenGymDataContainer> action)
    {
        NS_LOG_FUNCTION(this << action);
        Ptr<OpenGymDiscreteContainer> discrete = DynamicCast<OpenGymDiscreteContainer>(action);
        uint32_t nextChannel = discrete->GetValue();

        m_currentChannel = nextChannel;
        return true;
    }



    void
    GymActorCritic::CollectChannelOccupation(uint16_t chanId, uint32_t occupied)
    {
        NS_LOG_FUNCTION(this);
        m_channelOccupation= {0,0,0,0,0,0,0,0,0,0,0,0};
        m_channelOccupation[chanId]=occupied;
    }

    bool 
    GymActorCritic::CheckIfReady()
    {
        NS_LOG_FUNCTION(this);
        return m_channelOccupation.size() == m_channelNum;
    }

    void
    GymActorCritic::ClearObs()
    {
        NS_LOG_FUNCTION(this);
        m_channelOccupation.clear();
    }


    void 
    GymActorCritic::UpdateActorCriticWithRss(Ptr<GymActorCritic> entity, uint16_t channelId, double avgPower)
    {
        double threshold = -110;
        uint32_t busy =0;
        if(avgPower > threshold)
        {
            busy = 1;
        }
        entity->CollectChannelOccupation(channelId,busy);
        if(entity->CheckIfReady()){
            entity->Notify();
            entity->ClearObs();
        }
    }


    void
    GymActorCritic::UpdateActorCriticWithPacket(Ptr<GymActorCritic> entity, uint16_t channelId, uint32_t numberPacket)
    {
        uint32_t busy =0;
        if(numberPacket > 0)
        {
            busy = 2;
        }
        else{
            busy = 1;
        }
        entity->CollectChannelOccupation(channelId,busy);
       
        entity->Notify();
            
        
    }

}