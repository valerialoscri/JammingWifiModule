#include "my-gym-env-jammer.h"
#include "ns3/object.h"
#include "ns3/core-module.h"
#include "ns3/wifi-module.h"
#include "ns3/node-list.h"
#include "ns3/log.h"
#include <sstream>
#include <iostream>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("MyGymEnvJammer");

NS_OBJECT_ENSURE_REGISTERED (MyGymEnvJammer);

MyGymEnvJammer::MyGymEnvJammer()
{
    NS_LOG_FUNCTION(this);
    m_currentNode=0;
    m_currentChannel=0;
    m_collisionTh=22;
    m_channelNum =1;
    m_channelOccupation.clear();
}

MyGymEnvJammer::MyGymEnvJammer(uint32_t currentChannel, uint32_t channelNum )
{ 
     NS_LOG_FUNCTION(this);
     m_currentNode=0;
     m_currentChannel = currentChannel;
     m_collisionTh=22;
     m_channelNum= channelNum;
     m_channelOccupation.clear();
}

MyGymEnvJammer::~MyGymEnvJammer ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
MyGymEnvJammer::GetTypeId (void)
{
  static TypeId tid = TypeId ("MyGymEnvJammer")
    .SetParent<OpenGymEnv> ()
    .SetGroupName ("OpenGym")
    .AddConstructor<MyGymEnvJammer> ()
  ;
  return tid;
}

void
MyGymEnvJammer::DoDispose ()
{
  NS_LOG_FUNCTION (this);
}

Ptr<OpenGymSpace>
MyGymEnvJammer::GetActionSpace()
{
  NS_LOG_FUNCTION (this);
  Ptr<OpenGymDiscreteSpace> space = CreateObject <OpenGymDiscreteSpace> (m_channelNum);
  NS_LOG_UNCOND ("GetActionSpace:" << space);
  return space;
}

Ptr<OpenGymSpace>
MyGymEnvJammer:: GetObservationSpace()
{
    NS_LOG_FUNCTION (this);
    float low =0.0;
    float high = 1.0;
    std::vector<uint32_t> shape = {m_channelNum,};
    std::string dtype= TypeNameGet<uint32_t>();
    Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace>(low,high,shape,dtype);
    NS_LOG_UNCOND ("GetObservationsSpace: " << space);
    return space;
}

bool
MyGymEnvJammer::GetGameOver()
{
    NS_LOG_FUNCTION (this);
    bool isGameOver = false;

    uint32_t  collisionNum = 0;
    for(auto& v : m_collisions){
     collisionNum +=v;
    }

    if(collisionNum >= m_collisionTh){
        isGameOver=true;
    }

    NS_LOG_UNCOND ("MyGetGameOver: " << isGameOver);
    return isGameOver;
}

Ptr<OpenGymDataContainer>
MyGymEnvJammer::GetObservation()
{
    NS_LOG_FUNCTION (this);
    std::vector<uint32_t> shape = {m_channelNum,};
    Ptr<OpenGymBoxContainer<uint32_t> > box = CreateObject<OpenGymBoxContainer<uint32_t> >(shape);

    for(uint32_t i=0;i <m_channelOccupation.size(); i++){
        uint32_t value = m_channelOccupation.at(i);
        box->AddValue(value);
    }

    NS_LOG_UNCOND ("MyGetObservation: " << box);
    return box;
}

float
MyGymEnvJammer::GetReward()
{
    NS_LOG_FUNCTION (this);
    float reward = 1.0;
    if(m_channelOccupation.size()==0){
        return 0.0;
    }

    uint32_t occupied = m_channelOccupation.at(m_currentChannel);
    if(occupied ==1){
        m_collisions.erase(m_collisions.begin());
        m_collisions.push_back(0);
        
    }
    else{
        reward =-1.0;
        m_collisions.erase(m_collisions.begin());
        m_collisions.push_back(1);
    }

    NS_LOG_UNCOND ("MyGetReward: " << reward);
    return reward;
}

std::string
MyGymEnvJammer::GetExtraInfo(){
    NS_LOG_FUNCTION (this);
    std::string myInfo = "info";
    NS_LOG_UNCOND("MyGetExtraInfo: " << myInfo);
    return myInfo;
}

bool 
MyGymEnvJammer::ExecuteActions(Ptr<OpenGymDataContainer> action){
    NS_LOG_FUNCTION (this);
    Ptr<OpenGymDiscreteContainer> discrete = DynamicCast<OpenGymDiscreteContainer>(action);
    uint32_t nextChannel = discrete->GetValue();
    m_currentChannel = nextChannel;
    NS_LOG_UNCOND ("Current Channel: " << m_currentChannel);
    return true;
}

uint32_t 
MyGymEnvJammer::GetChoiseChannel(){
  return m_currentChannel;
}

void
MyGymEnvJammer::CollectChannelOccupation(uint16_t chanId,uint32_t occupied){
     NS_LOG_FUNCTION (this);
     m_channelOccupation.push_back(occupied);
}


bool
MyGymEnvJammer::CheckIfReady()
{
     NS_LOG_FUNCTION (this);
     return m_channelOccupation.size() == m_channelNum;
}

void
MyGymEnvJammer::ClearObs()
{
     NS_LOG_FUNCTION (this);
     m_channelOccupation.clear();
}

void
MyGymEnvJammer::PerformCca(Ptr<MyGymEnvJammer> entity, uint16_t channelId, double avgPower)
{
    double threshold = 80;
    uint32_t busy = 0;
    NS_LOG_FUNCTION (avgPower);
    if(avgPower < threshold){
      busy = 1;
    }
     NS_LOG_FUNCTION (busy);
    //NS_LOG_UNCOND("Channel: " << channelId << " CCA: " << busy << " RxPower: " << powerDbW);
    entity->CollectChannelOccupation(channelId, busy);
  if (entity->CheckIfReady()){
    entity->Notify();
    entity->ClearObs();
  }
}

void
MyGymEnvJammer::PerformJamming (Ptr<MyGymEnvJammer> entity, uint16_t channelId, double avgPower)
{
    double threshold = -100;
    uint32_t busy = 0;
    NS_LOG_FUNCTION (avgPower);
    if(avgPower > threshold){
      busy = 1;
    }
     NS_LOG_FUNCTION (busy);
    //NS_LOG_UNCOND("Channel: " << channelId << " CCA: " << busy << " RxPower: " << powerDbW);
    entity->CollectChannelOccupation(channelId, busy);
  if (entity->CheckIfReady()){
    entity->Notify();
    entity->ClearObs();
  }
}

}