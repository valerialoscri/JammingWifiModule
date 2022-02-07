#include "actor-critic-jammer.h"
#include "ns3/object.h"
#include "ns3/core-module.h"
#include "ns3/wifi-module.h"
#include "ns3/node-list.h"
#include "ns3/log.h"
#include <sstream>
#include <iostream>


namespace ns3 {

    NS_LOG_COMPONENT_DEFINE ("ActorCriticJammer");

    NS_OBJECT_ENSURE_REGISTERED (ActorCriticJammer);

    TypeId
    ActorCriticJammer::GetTypeId(void)
    {
        static TypeId tid = TypeId("ns3::ActorCriticJammer")
        .SetParent<Jammer> ()
        .AddConstructor <ActorCriticJammer>()
        .AddAttribute ("ActorCriticJammerJammerTxPower",
                   "Power to send jamming signal for constant jammer, in Watts.",
                   DoubleValue (0.001), // 0dBm
                   MakeDoubleAccessor (&ActorCriticJammer::SetTxPower,
                                       &ActorCriticJammer::GetTxPower),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("ActorCriticJammerJammingDuration",
                   "Jamming duration for constant jammer.",
                   TimeValue (MilliSeconds (15)),
                   MakeTimeAccessor (&ActorCriticJammer::SetJammingDuration,
                                     &ActorCriticJammer::GetJammingDuration),
                   MakeTimeChecker ())
    .AddAttribute ("ActorCriticJammerConstantInterval",
                   "Constant jammer jamming interval.",
                   TimeValue (MilliSeconds (0.1)),  // Set to 0 for continuous jamming
                   MakeTimeAccessor (&ActorCriticJammer::SetJammingInterval,
                                     &ActorCriticJammer::GetJammingInterval),
                   MakeTimeChecker ())
    .AddAttribute ("ActorCriticJammerMaxChannel",
                   "Constant jammer MaxChannel.",
                   UintegerValue (11),
                   MakeUintegerAccessor (&ActorCriticJammer::SetMaxChannel,
                                     &ActorCriticJammer::GetMaxChannel),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("ActorCriticJammerRxTimeout",
                   "Actor Critic jammer RX timeout.",
                   TimeValue (Seconds (0.4)),
                   MakeTimeAccessor (&ActorCriticJammer::SetRxTimeout,
                                     &ActorCriticJammer::GetRxTimeout),
                   MakeTimeChecker ())

        ;
     return tid;
    }

    ActorCriticJammer::ActorCriticJammer()
    {
        NS_LOG_FUNCTION(this);
        m_maxChannel = 12;
        m_already_begin = false;
        m_state_period = 0;
        m_numOfPktsReceived = 0;
        m_epoch=0;
        m_epochAttack=200;
        m_epochListen=0;
    }

    ActorCriticJammer::~ActorCriticJammer()
    {
        NS_LOG_FUNCTION(this);
    }

    void
    ActorCriticJammer::DoDispose()
    {
        NS_LOG_FUNCTION(this);
        m_jammingEvent.Cancel();
    }

    double 
    ActorCriticJammer::GetTxPower(void) const
    {
        return m_txPower;
    }

    uint16_t 
    ActorCriticJammer::GetMaxChannel(void) const
    {
        return m_maxChannel;
    }

    Time 
    ActorCriticJammer::GetJammingDuration(void) const
    {
        return m_jammingDuration;
    }

    Time 
    ActorCriticJammer::GetJammingInterval(void) const
    {
        return m_jammingInterval;
    }

    Time
    ActorCriticJammer::GetRxTimeout (void) const
    {
    NS_LOG_FUNCTION (this);
    return m_rxTimeout;
    }

    void
    ActorCriticJammer::SetUtility(Ptr<WirelessModuleUtility> utility)
    {
        NS_LOG_FUNCTION(this);
        NS_ASSERT(utility != NULL);
        m_utility = utility;
    }


    void
    ActorCriticJammer::SetTxPower(double power)
    {
        NS_LOG_FUNCTION(this);
        m_txPower = power;
    }

    void
    ActorCriticJammer::SetMaxChannel(uint16_t maxChannel)
    {
        NS_LOG_FUNCTION(this);
        m_maxChannel = maxChannel;
    }

    void 
    ActorCriticJammer::SetJammingDuration(Time duration)
    {
        NS_LOG_FUNCTION(this);
        m_jammingDuration = duration;
    }

    void
    ActorCriticJammer::SetJammingInterval(Time interval)
    {
        NS_LOG_FUNCTION(this);
        m_jammingInterval = interval;
    }

    void
    ActorCriticJammer::SetRxTimeout (Time rxTimeout)
    {
    NS_LOG_FUNCTION (this << rxTimeout);
    m_rxTimeout = rxTimeout;
    }

    


    void
    ActorCriticJammer::DoStopJamming(void)
    {
        NS_LOG_FUNCTION(this);
        m_jammingEvent.Cancel();
    }

    void
    ActorCriticJammer::DoJamming(void)
    {
        NS_LOG_FUNCTION(this);
        NS_ASSERT(m_utility != NULL);

        if(!IsJammerOn()) // check if jammer is on
        {
            NS_LOG_DEBUG("ActorCriticJammer: At Node" << GetId() << ",Jammer is OFF!");
            return ;
        }

        if(!m_already_begin){
        BeginActorCritic();
        m_already_begin = true;
        }

        m_rxTimeoutEvent.Cancel();
        m_rxTimeoutEvent = Simulator::Schedule(Seconds(0.3),&ActorCriticJammer::RxTimeoutHandler,this);
       
    }

    bool
    ActorCriticJammer::DoStartRxHandler(Ptr<Packet> packet, double startRss)
    {
        NS_LOG_FUNCTION(this << packet << startRss);
       // m_jammingEvent.Cancel();
        //m_jammingEvent = Simulator::Schedule(m_jammingInterval,&ActorCriticJammer::DoJamming,this);
        m_numOfPktsReceived = 0;
        NS_LOG_FUNCTION(packet->ToString());
        // 
        if (packet != NULL)
        {
              m_numOfPktsReceived++;
              NS_LOG_FUNCTION(this<<"packet number"<< m_numOfPktsReceived);
              
              
        }

        double actualPower = m_utility->SendJammingSignal(m_txPower, m_jammingDuration);
        if (actualPower != 0.0)
                {
                NS_LOG_DEBUG ("ActorCriticJammer:At Node #" << GetId () <<
                                ", Jamming signal sent with power = " << actualPower << " W");
                }
        else
                {
                NS_LOG_ERROR ("ActorCriticJammer:At Node #" << GetId () <<
                                ", Failed to send jamming signal!");
                }

         return false;
    }

    bool
    ActorCriticJammer::DoEndRxHandler(Ptr<Packet> packet, double averageRss)
    {
        NS_LOG_FUNCTION(this << packet << averageRss);
        
          // check if packet is valid, NULL means receive failed

        return false;
    }
    
    void
    ActorCriticJammer::DoEndTxHandler(Ptr<Packet> packet, double txPower)
    {
        NS_LOG_FUNCTION(this << packet << txPower);
        

    }

    void
    ActorCriticJammer::RxTimeoutHandler(void)
    {
        NS_LOG_FUNCTION(this);
        NS_ASSERT (m_utility != NULL);
        // after jamming change channel
        uint16_t currentChannel = m_utility->GetPhyLayerInfo ().currentChannel;
        uint16_t nextChannel = GetObservationsActorCritic();
        NS_LOG_FUNCTION(this << "numberPacket"<< m_numOfPktsReceived);
        m_numOfPktsReceived = 0 ;
        m_epoch++;
        NS_LOG_FUNCTION(this << "state"<< m_state_period);
        NS_LOG_FUNCTION(this << "epoch"<< m_epoch);
        if(m_state_period==0 && m_epochListen < m_epoch){
            //listen case
            m_state_period=1;
            m_epoch = 0; 
        }
        if(m_state_period==1 && m_epochAttack < m_epoch){
            m_state_period = 0;
         
            m_epoch=0;
        }
        NS_LOG_FUNCTION(this << "channel"<< nextChannel << "cuurentchannel"<<currentChannel);
        if(nextChannel!=currentChannel){
        m_utility->SwitchChannel(nextChannel);
        }

        //TODO change phase attack or listen
       
        //m_jammingEvent.Cancel();
        //m_jammingEvent=Simulator::Schedule(Seconds(1),&ActorCriticJammer::DoJamming,this);
    
         m_rxTimeoutEvent.Cancel();
         m_rxTimeoutEvent = Simulator::Schedule(Seconds(0.3),&ActorCriticJammer::RxTimeoutHandler,this);
       
    }


    void ActorCriticJammer::SetEnergySource(Ptr<EnergySource> source){}

    void ActorCriticJammer::DoStopJammer(void){}

    void ActorCriticJammer::Optimal(uint32_t channel){}
   

    void
    ActorCriticJammer::BeginActorCritic()
    {
        NS_LOG_FUNCTION(this);
        uint32_t openGymPort=5557;
        Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface> (openGymPort);
        m_myGymEnv = CreateObject<GymActorCritic>(m_utility->GetPhyLayerInfo().currentChannel,12);
        m_myGymEnv->SetOpenGymInterface(openGymInterface);
    }

    uint32_t 
    ActorCriticJammer::GetObservationsActorCritic()
    {
        //double rss = m_utility->GetRssinDbm();
        m_myGymEnv->UpdateActorCriticWithPacket(m_myGymEnv,m_utility->GetPhyLayerInfo ().currentChannel,m_numOfPktsReceived);
        uint32_t channel = m_myGymEnv->GetCurrentChannel();
        return  channel;

    }


}