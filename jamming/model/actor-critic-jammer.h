#ifndef ACTOR_CRITIC_JAMMER_H
#define ACTOR_CRITIC_JAMMER_H

#include "gym-actor-critic.h"
#include "jammer.h"
#include "ns3/nstime.h"
#include "ns3/event-id.h"
#include "wireless-module-utility.h"



namespace ns3 {

    class ActorCriticJammer: public Jammer

    {
        public: 
            static TypeId GetTypeId(void);
            ActorCriticJammer();
            virtual ~ActorCriticJammer();


            
            // Accessors

            double GetTxPower(void) const;
            uint16_t GetMaxChannel(void) const;
            Time GetJammingDuration(void) const;
            Time GetJammingInterval(void) const;
            Time GetRxTimeout (void) const;
            void SetUtility(Ptr<WirelessModuleUtility> utility);
            void SetTxPower(double power);
            void SetMaxChannel(uint16_t maxChannel);
            void SetJammingDuration(Time duration);
            void SetJammingInterval(Time interval);
            void SetRxTimeout (Time rxTimeout);




            // Jamming Function 

            virtual void DoStopJamming(void);
            virtual void DoJamming(void);
            virtual void DoDispose();
            virtual bool DoStartRxHandler(Ptr<Packet> packet, double startRss);
            virtual bool DoEndRxHandler(Ptr<Packet> packet, double averageRss);
            virtual void DoEndTxHandler(Ptr<Packet> packet, double txPower);
            void RxTimeoutHandler(void);
        
            // TODO Virtual function a implemter 
            void SetEnergySource(Ptr<EnergySource> source);

            void DoStopJammer(void);

            void Optimal(uint32_t channel);

            

            //Other Function

            void BeginActorCritic();
            uint32_t GetObservationsActorCritic();



        private:
            Ptr<WirelessModuleUtility> m_utility; // pointer to utility
            Ptr<EnergySource> m_source;           // pointer to energy source
            Time m_jammingInterval;       // jamming interval
            double m_txPower;                     // TX power
            Time m_jammingDuration;               // jamming duration
            EventId m_jammingEvent;               // jamming event     
            uint16_t m_maxChannel;
            bool m_jamming;
            Ptr<GymActorCritic> m_myGymEnv;
            EventId m_rxTimeoutEvent; 
            Time m_rxTimeout;
            bool m_already_begin;
            uint32_t m_numOfPktsReceived;
            bool m_state_period;  //state_period 0-listen,1-attack 
            uint32_t m_epochAttack;
            uint32_t m_epochListen;
            uint32_t m_epoch;
         
    };
}

#endif